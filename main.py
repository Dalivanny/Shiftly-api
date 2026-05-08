from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from ortools.sat.python import cp_model
import io
import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ScheduleRequest(BaseModel):
    employees: List[str]
    levels: List[str]
    days: List[str]
    shifts: List[str]
    closed_days: List[str]
    staff_requirements: Dict[str, Any]
    opening_requirements: Dict[str, Any]
    availability: List[List[List[int]]]
    rules: List[Dict[str, str]]

class PDFRequest(BaseModel):
    schedule: Dict[str, Dict[str, str]]
    staff: List[Dict[str, str]]
    days: List[str]
    shift_times: List[str]
    closed_days: List[str]
    shift_counts: Dict[str, int]
    rules: List[Dict[str, str]]
    week_start: Optional[str] = None
    show_level_divider: Optional[bool] = True

@app.get("/")
def root():
    return {"status": "Shiftly API running"}

@app.post("/generate")
def generate_schedule(data: ScheduleRequest):
    employees = data.employees
    levels = data.levels
    days = data.days
    shifts = data.shifts
    closed_days = data.closed_days
    availability = data.availability
    rules = data.rules

    num_employees = len(employees)
    num_days = len(days)
    num_shifts = len(shifts) if shifts else 2
    if availability and availability[0] and availability[0][0]:
        num_shifts = len(availability[0][0])

    model = cp_model.CpModel()

    shift_assigned = {}
    for e in range(num_employees):
        for d in range(num_days):
            for s in range(num_shifts):
                shift_assigned[(e, d, s)] = model.new_bool_var(f"shift_e{e}_d{d}_s{s}")

    for e in range(num_employees):
        for d in range(num_days):
            for s in range(num_shifts):
                if availability[e][d][s] == 0:
                    model.add(shift_assigned[(e, d, s)] == 0)

    for d, day in enumerate(days):
        if day in closed_days:
            for e in range(num_employees):
                for s in range(num_shifts):
                    model.add(shift_assigned[(e, d, s)] == 0)

    for e in range(num_employees):
        for d in range(num_days):
            model.add(sum(shift_assigned[(e, d, s)] for s in range(num_shifts)) <= 1)

    for d, day in enumerate(days):
        if day in closed_days:
            continue
        day_req = data.staff_requirements.get(day, {})
        min_s = day_req.get('min', 0)
        max_s = day_req.get('max', num_employees)
        total = sum(shift_assigned[(e, d, s)] for e in range(num_employees) for s in range(num_shifts))
        model.add(total >= min_s)
        model.add(total <= max_s)

    for d, day in enumerate(days):
        if day in closed_days:
            continue
        req_open = data.opening_requirements.get(day, 0)
        total_open = sum(shift_assigned[(e, d, 0)] for e in range(num_employees))
        model.add(total_open == req_open)

    for rule in rules:
        level1 = rule.get('level1')
        level2 = rule.get('level2')
        supervisors = [i for i, l in enumerate(levels) if l == level1]
        supervised = [i for i, l in enumerate(levels) if l == level2]
        for d in range(num_days):
            for e_n in supervised:
                works_today = model.new_bool_var(f"works_{e_n}_d{d}")
                model.add(sum(shift_assigned[(e_n, d, s)] for s in range(num_shifts)) >= 1).only_enforce_if(works_today)
                model.add(sum(shift_assigned[(e_n, d, s)] for s in range(num_shifts)) == 0).only_enforce_if(works_today.negated())
                supervisor_present = sum(shift_assigned[(e_s, d, s)] for e_s in supervisors for s in range(num_shifts))
                model.add(supervisor_present >= 1).only_enforce_if(works_today)

    available_this_week = [e for e in range(num_employees) if any(availability[e][d][s] == 1 for d in range(num_days) for s in range(num_shifts))]
    total_shifts_per = []
    for e in available_this_week:
        total = sum(shift_assigned[(e, d, s)] for d in range(num_days) for s in range(num_shifts))
        total_shifts_per.append(total)

    if total_shifts_per:
        min_shifts = model.new_int_var(0, 7, "min_shifts")
        model.add_min_equality(min_shifts, total_shifts_per)
        model.maximize(min_shifts)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30.0
    status = solver.solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"error": "No valid schedule found. Check availability and staffing rules."}

    result = {}
    shift_counts = {}
    for e in range(num_employees):
        name = employees[e]
        result[name] = {}
        shift_counts[name] = 0
        for d in range(num_days):
            day = days[d]
            assigned = None
            for s in range(num_shifts):
                if solver.value(shift_assigned[(e, d, s)]) == 1:
                    assigned = shifts[s]
            if day in closed_days:
                result[name][day] = "—"
            elif assigned:
                result[name][day] = assigned
                shift_counts[name] += 1
            else:
                available = any(availability[e][d][s] == 1 for s in range(num_shifts))
                result[name][day] = "OFF" if available else "N/A"

    return {"success": True, "schedule": result, "shift_counts": shift_counts, "employees": employees, "levels": levels}


def get_date_for_day(week_start_str, day_index):
    try:
        from datetime import datetime as dt, timedelta
        start = dt.strptime(week_start_str, '%Y-%m-%d')
        return (start + timedelta(days=day_index)).strftime('%d %b')
    except:
        return ''


@app.post("/generate-pdf")
def generate_pdf(req: PDFRequest):
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_RIGHT

    BG_DARK      = colors.HexColor('#0f0e0c')
    AMBER        = colors.HexColor('#e8a830')
    AMBER_LIGHT  = colors.HexColor('#faeeda')
    TEAL         = colors.HexColor('#1D9E75')
    TEAL_LIGHT   = colors.HexColor('#E1F5EE')
    PURPLE_LIGHT = colors.HexColor('#EEEDFE')
    PURPLE       = colors.HexColor('#534AB7')
    CORAL_LIGHT  = colors.HexColor('#FAECE7')
    CORAL        = colors.HexColor('#993C1D')
    BLUE_LIGHT   = colors.HexColor('#E8F4FD')
    BLUE         = colors.HexColor('#1A6FA8')
    PINK_LIGHT   = colors.HexColor('#FDE8F4')
    PINK         = colors.HexColor('#A81A6F')
    GREEN_LIGHT  = colors.HexColor('#E8FDE8')
    GREEN        = colors.HexColor('#1A8A1A')
    ROW_ALT      = colors.HexColor('#f7f5f0')
    ROW_WHITE    = colors.white
    BORDER       = colors.HexColor('#e0dcd4')
    TEXT_DARK    = colors.HexColor('#1a1916')
    TEXT_MID     = colors.HexColor('#5a5854')
    TEXT_LIGHT   = colors.HexColor('#9a9894')
    RED_LIGHT    = colors.HexColor('#FCEBEB')
    RED          = colors.HexColor('#A32D2D')
    CLOSED_BG    = colors.HexColor('#f5f0f0')
    CLOSED_TEXT  = colors.HexColor('#cc4444')

    # Color palette for shift times - cycles through if more than palette size
    SHIFT_PALETTE = [
        (AMBER_LIGHT, AMBER),
        (TEAL_LIGHT, TEAL),
        (PURPLE_LIGHT, PURPLE),
        (BLUE_LIGHT, BLUE),
        (PINK_LIGHT, PINK),
        (GREEN_LIGHT, GREEN),
        (CORAL_LIGHT, CORAL),
    ]

    def get_shift_color(shift_time, shift_times):
        try:
            idx = shift_times.index(shift_time)
            return SHIFT_PALETTE[idx % len(SHIFT_PALETTE)]
        except:
            return (AMBER_LIGHT, AMBER)

    def level_color(level):
        # Fixed colors for known levels
        known = {
            'Senior':    (AMBER_LIGHT, AMBER),
            'Junior':    (TEAL_LIGHT, TEAL),
            'New Staff': (PURPLE_LIGHT, PURPLE),
            'Trainee':   (CORAL_LIGHT, CORAL),
        }
        if level in known:
            return known[level]
        # For custom levels, cycle through palette based on hash of name
        palette = [
            (BLUE_LIGHT, BLUE),
            (PINK_LIGHT, PINK),
            (GREEN_LIGHT, GREEN),
            (AMBER_LIGHT, AMBER),
            (TEAL_LIGHT, TEAL),
            (PURPLE_LIGHT, PURPLE),
            (CORAL_LIGHT, CORAL),
        ]
        return palette[hash(level) % len(palette)]

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=landscape(A4),
        leftMargin=14*mm, rightMargin=14*mm,
        topMargin=14*mm, bottomMargin=14*mm,
    )
    story = []

    show_level_divider = req.show_level_divider if req.show_level_divider is not None else True

    # Header
    week_str = req.week_start or datetime.date.today().strftime('%d %b %Y')
    if req.week_start:
        try:
            from datetime import datetime as dt, timedelta
            ws = dt.strptime(req.week_start, '%Y-%m-%d')
            we = ws + timedelta(days=6)
            week_str = f"{ws.strftime('%d %b')} – {we.strftime('%d %b %Y')}"
        except:
            pass

    header_data = [[
        Paragraph('<b>Shiftly</b>', ParagraphStyle('logo', fontName='Helvetica-Bold', fontSize=18, textColor=AMBER)),
        Paragraph('Weekly Schedule', ParagraphStyle('rest', fontName='Helvetica-Bold', fontSize=13, textColor=TEXT_DARK)),
        Paragraph(f'Week of {week_str}<br/><font size=8 color=grey>Generated {datetime.date.today().strftime("%d %b %Y")}</font>',
            ParagraphStyle('wk', fontName='Helvetica', fontSize=10, textColor=TEXT_MID, alignment=TA_RIGHT)),
    ]]
    header_tbl = Table(header_data, colWidths=[30*mm, 180*mm, 59*mm])
    header_tbl.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LINEBELOW', (0,0), (-1,0), 0.5, BORDER),
        ('BOTTOMPADDING', (0,0), (-1,0), 10),
        ('TOPPADDING', (0,0), (-1,0), 4),
    ]))
    story.append(header_tbl)
    story.append(Spacer(1, 8))

    # Legend - dynamic based on shift times
    legend_items = [Paragraph('<b>Legend:</b>', ParagraphStyle('lg', fontName='Helvetica-Bold', fontSize=8, textColor=TEXT_MID))]
    legend_widths = [18*mm]
    for i, st in enumerate(req.shift_times):
        bg, fg = SHIFT_PALETTE[i % len(SHIFT_PALETTE)]
        legend_items.append(Paragraph(f'{st}', ParagraphStyle(f'lg{i}', fontName='Helvetica-Bold', fontSize=8, textColor=fg)))
        legend_widths.append(22*mm)
    legend_items.append(Paragraph('OFF = available, not scheduled', ParagraphStyle('lgoff', fontName='Helvetica', fontSize=8, textColor=TEXT_LIGHT)))
    legend_widths.append(55*mm)
    legend_items.append(Paragraph('N/A = not available', ParagraphStyle('lgna', fontName='Helvetica', fontSize=8, textColor=RED)))
    legend_widths.append(35*mm)
    legend_items.append(Paragraph('CLOSED = closed day', ParagraphStyle('lgcl', fontName='Helvetica', fontSize=8, textColor=CLOSED_TEXT)))
    legend_widths.append(35*mm)

    # Fill remaining width
    total_used = sum(legend_widths)
    page_w = landscape(A4)[0] - 28*mm
    if total_used < page_w:
        legend_widths.append(page_w - total_used)
        legend_items.append(Paragraph('', ParagraphStyle('lgx', fontName='Helvetica', fontSize=8)))

    legend_tbl = Table([legend_items], colWidths=legend_widths)
    legend_tbl.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(legend_tbl)
    story.append(Spacer(1, 4))

    # Schedule table
    col_w  = 19*mm
    name_w = 52*mm  # wider since no level column

    # Build date row and day row separately
    dates_row = ['', '']  # name + spacer
    days_row  = ['Name', '']
    for di, day in enumerate(req.days):
        date_str = get_date_for_day(req.week_start, di) if req.week_start else ''
        dates_row.append(date_str)
        days_row.append(day[:3])
    dates_row.append('')
    days_row.append('Total')

    col_widths = [name_w, 0] + [col_w] * len(req.days) + [14*mm]

    sched_data = [dates_row, days_row]
    row_styles = []
    r = 2

    # Get all unique levels, starting with known ones then any custom ones
    known_levels = ['Senior', 'Junior', 'New Staff', 'Trainee']
    all_levels_in_staff = []
    for s in req.staff:
        lv = s.get('level', '')
        if lv and lv not in all_levels_in_staff:
            all_levels_in_staff.append(lv)
    level_order = known_levels + [l for l in all_levels_in_staff if l not in known_levels]

    for level in level_order:
        group = [s for s in req.staff if s.get('level') == level]
        if not group:
            continue

        if show_level_divider:
            divider = [level.upper()] + [''] * (len(req.days) + 2)
            sched_data.append(divider)
            row_styles += [
                ('SPAN', (0, r), (-1, r)),
                ('BACKGROUND', (0, r), (-1, r), colors.HexColor('#f0ece4')),
                ('TEXTCOLOR', (0, r), (-1, r), TEXT_LIGHT),
                ('FONTNAME', (0, r), (-1, r), 'Helvetica-Bold'),
                ('FONTSIZE', (0, r), (-1, r), 7.5),
                ('TOPPADDING', (0, r), (-1, r), 5),
                ('BOTTOMPADDING', (0, r), (-1, r), 4),
            ]
            r += 1

        for s in group:
            name = s.get('name', '')
            count = req.shift_counts.get(name, 0)
            row = [name, '']
            for day in req.days:
                val = req.schedule.get(name, {}).get(day, 'N/A')
                row.append(val)
            row.append(str(count))
            sched_data.append(row)

            row_bg = ROW_WHITE if r % 2 == 0 else ROW_ALT
            row_styles += [
                ('BACKGROUND', (0, r), (-1, r), row_bg),
                ('FONTNAME', (0, r), (0, r), 'Helvetica-Bold'),
                ('FONTSIZE', (0, r), (0, r), 9),
            ]

            for di, day in enumerate(req.days):
                col = di + 2
                val = req.schedule.get(name, {}).get(day, 'N/A')
                if day in req.closed_days:
                    row_styles += [
                        ('BACKGROUND', (col, r), (col, r), ROW_WHITE if r % 2 == 0 else ROW_ALT),
                        ('TEXTCOLOR', (col, r), (col, r), colors.HexColor('#cccccc')),
                    ]
                elif val == 'N/A':
                    row_styles += [
                        ('BACKGROUND', (col, r), (col, r), RED_LIGHT),
                        ('TEXTCOLOR', (col, r), (col, r), RED),
                    ]
                elif val == 'OFF':
                    row_styles.append(('TEXTCOLOR', (col, r), (col, r), TEXT_LIGHT))
                else:
                    sc = get_shift_color(val, req.shift_times)
                    row_styles += [
                        ('BACKGROUND', (col, r), (col, r), sc[0]),
                        ('TEXTCOLOR', (col, r), (col, r), sc[1]),
                        ('FONTNAME', (col, r), (col, r), 'Helvetica-Bold'),
                    ]
            r += 1

    # Now handle closed day columns - merge all cells in closed day columns
    for di, day in enumerate(req.days):
        if day in req.closed_days:
            col = di + 2
            # Style the day header - same dark background as other days
            row_styles += [
                    ('BACKGROUND', (col, 1), (col, 1), BG_DARK),
                    ('TEXTCOLOR', (col, 1), (col, 1), AMBER),
                ]

    base_style = TableStyle([
        # Date row (row 0)
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e1c19')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#888480')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,0), 7.5),
        ('ALIGN', (0,0), (-1,0), 'CENTER'),
        ('TOPPADDING', (0,0), (-1,0), 4),
        ('BOTTOMPADDING', (0,0), (-1,0), 4),
        # Day name row (row 1)
        ('BACKGROUND', (0,1), (-1,1), BG_DARK),
        ('TEXTCOLOR', (0,1), (-1,1), AMBER),
        ('FONTNAME', (0,1), (-1,1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,1), (-1,1), 8),
        ('ALIGN', (0,1), (-1,1), 'CENTER'),
        ('TOPPADDING', (0,1), (-1,1), 6),
        ('BOTTOMPADDING', (0,1), (-1,1), 6),
        # Name col
        ('ALIGN', (0,2), (0,-1), 'LEFT'),
        ('TEXTCOLOR', (0,1), (0,1), colors.HexColor('#c8c4bc')),
        # Day cols center
        ('ALIGN', (2,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,2), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,2), (-1,-1), 8.5),
        ('TEXTCOLOR', (0,2), (0,-1), TEXT_DARK),
        ('TEXTCOLOR', (2,2), (-1,-1), TEXT_MID),
        ('TOPPADDING', (0,2), (-1,-1), 6),
        ('BOTTOMPADDING', (0,2), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ('GRID', (0,0), (-1,-1), 0.3, BORDER),
        # Hide the empty second column
        ('COLBACKGROUND', (1,0), (1,-1), colors.white),
    ] + row_styles)

    sched_table = Table(sched_data, colWidths=col_widths, style=base_style, repeatRows=2)
    story.append(sched_table)
    story.append(Spacer(1, 14))

    # Fairness table
    max_shifts = max(req.shift_counts.values()) if req.shift_counts else 1
    fair_data = [['Employee', 'Shifts', 'Distribution']]
    fair_styles = []
    for i, s in enumerate(req.staff):
        name = s.get('name', '')
        level = s.get('level', '')
        count = req.shift_counts.get(name, 0)
        bar = '█' * count + '░' * (max_shifts - count)
        fair_data.append([name, str(count), bar])
        lbg, ltxt = level_color(level)
        ri = i + 1
        fair_styles += [
            ('BACKGROUND', (0, ri), (0, ri), lbg),
            ('TEXTCOLOR', (0, ri), (0, ri), ltxt),
            ('FONTNAME', (0, ri), (0, ri), 'Helvetica-Bold'),
        ]

    fair_style = TableStyle([
        ('BACKGROUND', (0,0), (-1,0), BG_DARK),
        ('TEXTCOLOR', (0,0), (-1,0), AMBER),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 8),
        ('FONTSIZE', (0,1), (-1,-1), 8.5),
        ('TEXTCOLOR', (0,1), (0,-1), TEXT_DARK),
        ('GRID', (0,0), (-1,-1), 0.3, BORDER),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [ROW_WHITE, ROW_ALT]),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ] + fair_styles)
    fair_table = Table(fair_data, colWidths=[40*mm, 20*mm, 60*mm], style=fair_style)

    # Rules table
    rules_data = [['#', 'Rule']]
    for i, rule in enumerate(req.rules):
        rules_data.append([str(i+1), f"When a {rule.get('level2','')} is scheduled, at least one {rule.get('level1','')} must also work that day."])
    rules_style = TableStyle([
        ('BACKGROUND', (0,0), (-1,0), BG_DARK),
        ('TEXTCOLOR', (0,0), (-1,0), AMBER),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 8),
        ('FONTNAME', (0,1), (0,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,1), (-1,-1), 8.5),
        ('TEXTCOLOR', (0,1), (0,-1), TEXT_DARK),
        ('TEXTCOLOR', (1,1), (1,-1), TEXT_MID),
        ('GRID', (0,0), (-1,-1), 0.3, BORDER),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [ROW_WHITE, ROW_ALT]),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ])
    rules_table = Table(rules_data, colWidths=[12*mm, 140*mm], style=rules_style)

    bottom_data = [[fair_table, rules_table]]
    bottom_tbl = Table(bottom_data, colWidths=[130*mm, 139*mm])
    bottom_tbl.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (1,0), (1,0), 14),
    ]))
    story.append(Paragraph('FAIRNESS & RULES', ParagraphStyle('sec', fontName='Helvetica-Bold', fontSize=9, textColor=TEXT_LIGHT, spaceBefore=4, spaceAfter=6)))
    story.append(bottom_tbl)
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        'Generated by Shiftly · All supervision rules verified · Powered by OR-Tools',
        ParagraphStyle('footer', fontName='Helvetica', fontSize=8, textColor=TEXT_LIGHT, alignment=TA_CENTER)
    ))

    doc.build(story)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type='application/pdf',
        headers={'Content-Disposition': 'attachment; filename="shiftly-schedule.pdf"'}
    )