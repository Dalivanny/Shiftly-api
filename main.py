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
    num_shifts = len(shifts)

    model = cp_model.CpModel()

    shift_assigned = {}
    for e in range(num_employees):
        for d in range(num_days):
            for s in range(num_shifts):
                shift_assigned[(e, d, s)] = model.new_bool_var(f"shift_e{e}_d{d}_s{s}")

    # RULE 1: Respect availability
    for e in range(num_employees):
        for d in range(num_days):
            for s in range(num_shifts):
                if availability[e][d][s] == 0:
                    model.add(shift_assigned[(e, d, s)] == 0)

    # RULE 2: Closed days
    for d, day in enumerate(days):
        if day in closed_days:
            for e in range(num_employees):
                for s in range(num_shifts):
                    model.add(shift_assigned[(e, d, s)] == 0)

    # RULE 3: No double shifts
    for e in range(num_employees):
        for d in range(num_days):
            model.add(sum(shift_assigned[(e, d, s)] for s in range(num_shifts)) <= 1)

    # RULE 4: Min/max staff per day
    for d, day in enumerate(days):
        if day in closed_days:
            continue
        day_req = data.staff_requirements.get(day, {})
        min_s = day_req.get('min', 0)
        max_s = day_req.get('max', num_employees)
        total = sum(
            shift_assigned[(e, d, s)]
            for e in range(num_employees)
            for s in range(num_shifts)
        )
        model.add(total >= min_s)
        model.add(total <= max_s)

    # RULE 5: Opening staff per day
    for d, day in enumerate(days):
        if day in closed_days:
            continue
        req_open = data.opening_requirements.get(day, 0)
        total_open = sum(shift_assigned[(e, d, 0)] for e in range(num_employees))
        model.add(total_open == req_open)

    # SPECIAL RULES: supervision
    for rule in rules:
        level1 = rule.get('level1')
        level2 = rule.get('level2')
        supervisors = [i for i, l in enumerate(levels) if l == level1]
        supervised = [i for i, l in enumerate(levels) if l == level2]

        for d in range(num_days):
            for e_n in supervised:
                works_today = model.new_bool_var(f"works_{e_n}_d{d}")
                model.add(
                    sum(shift_assigned[(e_n, d, s)] for s in range(num_shifts)) >= 1
                ).only_enforce_if(works_today)
                model.add(
                    sum(shift_assigned[(e_n, d, s)] for s in range(num_shifts)) == 0
                ).only_enforce_if(works_today.negated())
                supervisor_present = sum(
                    shift_assigned[(e_s, d, s)]
                    for e_s in supervisors
                    for s in range(num_shifts)
                )
                model.add(supervisor_present >= 1).only_enforce_if(works_today)

    # FAIRNESS
    available_this_week = [
        e for e in range(num_employees)
        if any(availability[e][d][s] == 1 for d in range(num_days) for s in range(num_shifts))
    ]

    total_shifts_per = []
    for e in available_this_week:
        total = sum(
            shift_assigned[(e, d, s)]
            for d in range(num_days)
            for s in range(num_shifts)
        )
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
                result[name][day] = "CLOSED"
            elif assigned:
                result[name][day] = assigned
                shift_counts[name] += 1
            else:
                available = any(availability[e][d][s] == 1 for s in range(num_shifts))
                result[name][day] = "OFF" if available else "N/A"

    return {
        "success": True,
        "schedule": result,
        "shift_counts": shift_counts,
        "employees": employees,
        "levels": levels,
    }

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
    ROW_ALT      = colors.HexColor('#f7f5f0')
    ROW_WHITE    = colors.white
    BORDER       = colors.HexColor('#e0dcd4')
    TEXT_DARK    = colors.HexColor('#1a1916')
    TEXT_MID     = colors.HexColor('#5a5854')
    TEXT_LIGHT   = colors.HexColor('#9a9894')
    RED_LIGHT    = colors.HexColor('#FCEBEB')
    RED          = colors.HexColor('#A32D2D')

    def level_color(level):
        return {
            'Senior':    (AMBER_LIGHT, AMBER),
            'Junior':    (TEAL_LIGHT, TEAL),
            'New Staff': (PURPLE_LIGHT, PURPLE),
            'Trainee':   (CORAL_LIGHT, CORAL),
        }.get(level, (ROW_WHITE, TEXT_MID))

    def shift_color(val, shift_times):
        if val == shift_times[0]:
            return (AMBER_LIGHT, AMBER)
        if len(shift_times) > 1 and val == shift_times[1]:
            return (TEAL_LIGHT, TEAL)
        return None

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        leftMargin=14*mm, rightMargin=14*mm,
        topMargin=14*mm, bottomMargin=14*mm,
    )
    story = []

    # Header
    week_str = req.week_start or datetime.date.today().strftime('%d %b %Y')
    header_data = [[
        Paragraph('<b>Shiftly</b>', ParagraphStyle('logo', fontName='Helvetica-Bold', fontSize=18, textColor=AMBER)),
        Paragraph('Weekly Schedule', ParagraphStyle('rest', fontName='Helvetica-Bold', fontSize=13, textColor=TEXT_DARK)),
        Paragraph(
            f'Week of {week_str}<br/><font size=8 color=grey>Generated {datetime.date.today().strftime("%d %b %Y")}</font>',
            ParagraphStyle('wk', fontName='Helvetica', fontSize=10, textColor=TEXT_MID, alignment=TA_RIGHT)
        ),
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

    # Legend
    legend_data = [[
        Paragraph('<b>Legend:</b>', ParagraphStyle('lg', fontName='Helvetica-Bold', fontSize=8, textColor=TEXT_MID)),
        Paragraph(f'{req.shift_times[0]} Opening', ParagraphStyle('lg2', fontName='Helvetica', fontSize=8, textColor=AMBER)),
        Paragraph(f'{req.shift_times[1] if len(req.shift_times) > 1 else ""} Closing', ParagraphStyle('lg3', fontName='Helvetica', fontSize=8, textColor=TEAL)),
        Paragraph('OFF = available, not scheduled', ParagraphStyle('lg4', fontName='Helvetica', fontSize=8, textColor=TEXT_LIGHT)),
        Paragraph('N/A = not available', ParagraphStyle('lg5', fontName='Helvetica', fontSize=8, textColor=RED)),
        Paragraph('— = restaurant closed', ParagraphStyle('lg6', fontName='Helvetica', fontSize=8, textColor=TEXT_LIGHT)),
    ]]
    legend_tbl = Table(legend_data, colWidths=[18*mm, 30*mm, 30*mm, 60*mm, 35*mm, 96*mm])
    legend_tbl.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(legend_tbl)
    story.append(Spacer(1, 4))

    # Schedule table
    col_w  = 19*mm
    name_w = 28*mm
    level_w = 24*mm

    header1 = ['', ''] + [d[:3] for d in req.days] + ['']
    header2 = ['Name', 'Level'] + ['' for _ in req.days] + ['Total']
    sched_data = [header1, header2]
    row_styles = []
    r = 2

    level_order = ['Senior', 'Junior', 'New Staff', 'Trainee']
    for level in level_order:
        group = [s for s in req.staff if s.get('level') == level]
        if not group:
            continue

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
            row = [name, level]
            for day in req.days:
                val = req.schedule.get(name, {}).get(day, 'N/A')
                row.append(val)
            row.append(str(count))
            sched_data.append(row)

            lbg, ltxt = level_color(level)
            row_bg = ROW_WHITE if r % 2 == 0 else ROW_ALT
            row_styles += [
                ('BACKGROUND', (0, r), (-1, r), row_bg),
                ('FONTNAME', (0, r), (0, r), 'Helvetica-Bold'),
                ('BACKGROUND', (1, r), (1, r), lbg),
                ('TEXTCOLOR', (1, r), (1, r), ltxt),
                ('FONTNAME', (1, r), (1, r), 'Helvetica-Bold'),
                ('FONTSIZE', (1, r), (1, r), 7.5),
            ]

            for di, day in enumerate(req.days):
                col = di + 2
                val = req.schedule.get(name, {}).get(day, 'N/A')
                if day in req.closed_days:
                    row_styles.append(('TEXTCOLOR', (col, r), (col, r), TEXT_LIGHT))
                elif val == 'N/A':
                    row_styles += [
                        ('BACKGROUND', (col, r), (col, r), RED_LIGHT),
                        ('TEXTCOLOR', (col, r), (col, r), RED),
                    ]
                elif val == 'OFF':
                    row_styles.append(('TEXTCOLOR', (col, r), (col, r), TEXT_LIGHT))
                else:
                    sc = shift_color(val, req.shift_times)
                    if sc:
                        row_styles += [
                            ('BACKGROUND', (col, r), (col, r), sc[0]),
                            ('TEXTCOLOR', (col, r), (col, r), sc[1]),
                            ('FONTNAME', (col, r), (col, r), 'Helvetica-Bold'),
                        ]
            r += 1

    col_widths = [name_w, level_w] + [col_w] * len(req.days) + [14*mm]
    base_style = TableStyle([
        ('BACKGROUND', (0,0), (-1,0), BG_DARK),
        ('TEXTCOLOR', (0,0), (-1,0), AMBER),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 8),
        ('ALIGN', (0,0), (-1,0), 'CENTER'),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor('#1e1c19')),
        ('TEXTCOLOR', (0,1), (-1,1), colors.HexColor('#c8c4bc')),
        ('FONTNAME', (0,1), (-1,1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,1), 8),
        ('ALIGN', (2,0), (-1,-1), 'CENTER'),
        ('ALIGN', (0,2), (0,-1), 'LEFT'),
        ('FONTNAME', (0,2), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,2), (-1,-1), 8.5),
        ('TEXTCOLOR', (0,2), (0,-1), TEXT_DARK),
        ('TEXTCOLOR', (2,2), (-1,-1), TEXT_MID),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ('GRID', (0,0), (-1,-1), 0.3, BORDER),
        ('LINEBELOW', (0,1), (-1,1), 0.5, colors.HexColor('#555')),
    ] + row_styles)

    sched_table = Table(sched_data, colWidths=col_widths, style=base_style, repeatRows=2)
    story.append(sched_table)
    story.append(Spacer(1, 14))

    # Bottom section: fairness + rules
    max_shifts = max(req.shift_counts.values()) if req.shift_counts else 1
    fair_data = [['Employee', 'Level', 'Shifts', 'Distribution']]
    fair_styles = []
    for i, s in enumerate(req.staff):
        name = s.get('name', '')
        level = s.get('level', '')
        count = req.shift_counts.get(name, 0)
        bar = '█' * count + '░' * (max_shifts - count)
        fair_data.append([name, level, str(count), bar])
        lbg, ltxt = level_color(level)
        ri = i + 1
        fair_styles += [
            ('BACKGROUND', (1, ri), (1, ri), lbg),
            ('TEXTCOLOR', (1, ri), (1, ri), ltxt),
            ('FONTNAME', (1, ri), (1, ri), 'Helvetica-Bold'),
            ('FONTSIZE', (1, ri), (1, ri), 7.5),
        ]

    fair_style = TableStyle([
        ('BACKGROUND', (0,0), (-1,0), BG_DARK),
        ('TEXTCOLOR', (0,0), (-1,0), AMBER),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 8),
        ('FONTNAME', (0,1), (0,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,1), (-1,-1), 8.5),
        ('TEXTCOLOR', (0,1), (0,-1), TEXT_DARK),
        ('GRID', (0,0), (-1,-1), 0.3, BORDER),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [ROW_WHITE, ROW_ALT]),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ] + fair_styles)

    fair_table = Table(fair_data, colWidths=[40*mm, 32*mm, 20*mm, 55*mm], style=fair_style)

    rules_data = [['#', 'Rule']]
    for i, rule in enumerate(req.rules):
        rules_data.append([
            str(i+1),
            f"When a {rule.get('level2','')} is scheduled, at least one {rule.get('level1','')} must also work that day."
        ])

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
    bottom_tbl = Table(bottom_data, colWidths=[155*mm, 114*mm])
    bottom_tbl.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (1,0), (1,0), 14),
    ]))
    story.append(Paragraph('FAIRNESS SUMMARY', ParagraphStyle('sec', fontName='Helvetica-Bold', fontSize=9, textColor=TEXT_LIGHT, spaceBefore=4, spaceAfter=6)))
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