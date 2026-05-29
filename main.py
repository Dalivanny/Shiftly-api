from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from ortools.sat.python import cp_model
import io
import datetime
import httpx
import os

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
    employee_roles: List[List[str]]  # roles each employee can do
    days: List[str]
    shifts: List[str]
    closed_days: List[str]
    shift_staff_requirements: Dict[str, Dict[str, int]]
    role_requirements: Dict[str, Dict[str, int]]  # { "Bartender": { "Monday": 2 } }
    availability: List[List[List[int]]]
    rules: List[Dict[str, str]]

class PDFRequest(BaseModel):
    schedule: Dict[str, Dict[str, str]]
    role_assignments: Optional[Dict[str, Dict[str, str]]] = None  # { "Haylee": { "Monday": "Bartender" } }
    staff: List[Dict[str, str]]
    days: List[str]
    shift_times: List[str]
    closed_days: List[str]
    shift_counts: Dict[str, int]
    rules: List[Dict[str, str]]
    week_start: Optional[str] = None
    show_level_divider: Optional[bool] = True
    all_roles: Optional[List[str]] = None  # all roles for legend

@app.get("/")
def root():
    return {"status": "Shiftly API running"}

@app.post("/notify-manager")
async def notify_manager(data: dict):
    manager_email = data.get('manager_email')
    team = data.get('team')
    week_start = data.get('week_start')
    staff_names = data.get('staff_names', [])
    resend_api_key = os.environ.get('RESEND_API_KEY')

    if not manager_email or not resend_api_key:
        return {"error": "Missing email or API key"}

    html = f"""
    <div style="font-family: Arial, sans-serif; max-width: 480px; margin: 0 auto; background: #ffffff;">
        <div style="background: #0f0e0c; padding: 20px 24px; border-radius: 8px 8px 0 0;">
            <span style="font-family: Georgia, serif; font-size: 20px;">
                <span style="color: #e8a830;">shift</span><span style="color: #ffffff;">ly</span>
            </span>
        </div>
        <div style="padding: 24px; border: 1px solid #e5e5e5; border-top: none; border-radius: 0 0 8px 8px;">
            <p style="margin: 0 0 16px; color: #333; font-size: 15px;">Hi,</p>
            <p style="margin: 0 0 16px; color: #333; font-size: 15px;">
                All staff from <strong>{team}</strong> have submitted their availability for the week of <strong>{week_start}</strong>.
            </p>
            <p style="margin: 0 0 8px; color: #666; font-size: 13px;">Staff who submitted:</p>
            <p style="margin: 0 0 20px; color: #333; font-size: 14px;">{', '.join(staff_names)}</p>
            <a href="https://viashiftly.com/dashboard"
            style="background: #e8a830; color: #0f0e0c; padding: 10px 20px; text-decoration: none; border-radius: 6px; font-weight: bold; font-size: 14px; display: inline-block;">
                Generate schedule →
            </a>
            <p style="margin: 24px 0 0; color: #999; font-size: 12px;">— Shiftly · viashiftly.com</p>
        </div>
    </div>
    """

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {resend_api_key}", "Content-Type": "application/json"},
                json={
                    "from": "Shiftly <hello@viashiftly.com>",
                    "to": [manager_email],
                    "subject": f"✓ All staff submitted — {team} week of {week_start}",
                    "html": html
                }
            )
            if response.status_code == 200:
                return {"success": True}
            else:
                return {"error": response.text}
    except Exception as e:
        return {"error": str(e)}


def get_abbrev(role, all_roles):
    others = [r for r in all_roles if r != role]
    a1 = role[0].upper()
    if a1 not in [r[0].upper() for r in others]:
        return a1
    a2 = role[:2].upper()
    if a2 not in [r[:2].upper() for r in others]:
        return a2
    return role[:3].upper()


@app.post("/generate")
def generate_schedule(data: ScheduleRequest):
    employees = data.employees
    levels = data.levels
    employee_roles = data.employee_roles
    days = data.days
    shifts = data.shifts
    closed_days = data.closed_days
    availability = data.availability
    rules = data.rules
    shift_staff_reqs = data.shift_staff_requirements
    role_reqs = data.role_requirements  # { "Bartender": { "Monday": 2 } }

    num_employees = len(employees)
    num_days = len(days)
    num_shifts = len(shifts) if shifts else 2
    if availability and len(availability) > 0 and len(availability[0]) > 0 and len(availability[0][0]) > 0:
        num_shifts = min(len(shifts), len(availability[0][0]))

    # Get all unique roles
    all_roles = list({r for roles in employee_roles for r in roles})
    num_roles = len(all_roles)
    role_index = {r: i for i, r in enumerate(all_roles)}

    model = cp_model.CpModel()

    # shift_assigned[e, d, s] = 1 if employee e works shift s on day d
    shift_assigned = {}
    for e in range(num_employees):
        for d in range(num_days):
            for s in range(num_shifts):
                shift_assigned[(e, d, s)] = model.new_bool_var(f"shift_e{e}_d{d}_s{s}")

    # role_assigned[e, d, r] = 1 if employee e is assigned role r on day d
    role_assigned = {}
    if num_roles > 0:
        for e in range(num_employees):
            for d in range(num_days):
                for r in range(num_roles):
                    role_assigned[(e, d, r)] = model.new_bool_var(f"role_e{e}_d{d}_r{r}")

    # RULE: Respect availability
    for e in range(num_employees):
        for d in range(num_days):
            for s in range(num_shifts):
                avail_val = availability[e][d][s] if s < len(availability[e][d]) else 0
                if avail_val == 0:
                    model.add(shift_assigned[(e, d, s)] == 0)

    # RULE: Closed days
    for d, day in enumerate(days):
        if day in closed_days:
            for e in range(num_employees):
                for s in range(num_shifts):
                    model.add(shift_assigned[(e, d, s)] == 0)

    # RULE: No double shifts
    for e in range(num_employees):
        for d in range(num_days):
            model.add(sum(shift_assigned[(e, d, s)] for s in range(num_shifts)) <= 1)

    # RULE: Exact staff per shift per day
    for d, day in enumerate(days):
        if day in closed_days:
            continue
        day_reqs = shift_staff_reqs.get(day, {})
        for s, shift_time in enumerate(shifts):
            if s >= num_shifts:
                continue
            required = day_reqs.get(shift_time)
            if required is not None and required > 0:
                total = sum(shift_assigned[(e, d, s)] for e in range(num_employees))
                model.add(total == required)
            elif required == 0 or shift_time not in day_reqs:
                for e in range(num_employees):
                    model.add(shift_assigned[(e, d, s)] == 0)

    # RULE: Role constraints (only if roles exist)
    if num_roles > 0:
        for e in range(num_employees):
            for d in range(num_days):
                works_today = sum(shift_assigned[(e, d, s)] for s in range(num_shifts))

                # If working, must be assigned exactly one role (if they have roles)
                emp_roles = employee_roles[e] if e < len(employee_roles) else []
                emp_role_indices = [role_index[r] for r in emp_roles if r in role_index]

                if emp_role_indices:
                    # Can only be assigned roles they can do
                    for r in range(num_roles):
                        if r not in emp_role_indices:
                            model.add(role_assigned[(e, d, r)] == 0)

                    # If working → exactly one role assigned
                    total_roles = sum(role_assigned[(e, d, r)] for r in range(num_roles))
                    model.add(total_roles == works_today)
                else:
                    # No roles defined for this employee — no role assignment
                    for r in range(num_roles):
                        model.add(role_assigned[(e, d, r)] == 0)

        # RULE: Exact role requirements per day
        for role_name, day_counts in role_reqs.items():
            if role_name not in role_index:
                continue
            ri = role_index[role_name]
            for d, day in enumerate(days):
                if day in closed_days:
                    continue
                required = day_counts.get(day, 0)
                if required > 0:
                    total = sum(role_assigned[(e, d, ri)] for e in range(num_employees))
                    model.add(total == required)

    # RULE: Supervision
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

    # FAIRNESS
    available_this_week = [
        e for e in range(num_employees)
        if any(s < len(availability[e][d]) and availability[e][d][s] == 1 for d in range(num_days) for s in range(num_shifts))
    ]
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
        issues = []
        for d, day in enumerate(days):
            if day in closed_days:
                continue
            day_reqs = shift_staff_reqs.get(day, {})
            for s, shift_time in enumerate(shifts):
                if s >= num_shifts:
                    continue
                required = day_reqs.get(shift_time, 0)
                if required > 0:
                    available_count = sum(1 for e in range(num_employees) if s < len(availability[e][d]) and availability[e][d][s] == 1)
                    if available_count < required:
                        issues.append(f"{day} {shift_time}: only {available_count} available but {required} needed")
        # Check role requirements
        for role_name, day_counts in role_reqs.items():
            if role_name not in role_index:
                continue
            ri = role_index[role_name]
            for d, day in enumerate(days):
                if day in closed_days:
                    continue
                required = day_counts.get(day, 0)
                if required > 0:
                    capable = sum(1 for e in range(num_employees) if role_name in (employee_roles[e] if e < len(employee_roles) else []))
                    if capable < required:
                        issues.append(f"{day} {role_name}: only {capable} staff can do this role but {required} needed")
        if issues:
            return {"error": "Could not generate schedule:\n• " + "\n• ".join(issues)}
        return {"error": "Could not generate schedule. Check that enough staff are available to meet your requirements."}

    result = {}
    shift_counts = {}
    role_assignments = {}  # { employee_name: { day: role_name } }

    for e in range(num_employees):
        name = employees[e]
        result[name] = {}
        shift_counts[name] = 0
        role_assignments[name] = {}
        for d in range(num_days):
            day = days[d]
            assigned_shift = None
            for s in range(num_shifts):
                if solver.value(shift_assigned[(e, d, s)]) == 1:
                    assigned_shift = shifts[s]

            # Get role assignment
            assigned_role = None
            if num_roles > 0 and assigned_shift:
                for r in range(num_roles):
                    if solver.value(role_assigned[(e, d, r)]) == 1:
                        assigned_role = all_roles[r]
                        break

            if day in closed_days:
                result[name][day] = "—"
            elif assigned_shift:
                result[name][day] = assigned_shift
                shift_counts[name] += 1
                if assigned_role:
                    role_assignments[name][day] = assigned_role
            else:
                available = any(s < len(availability[e][d]) and availability[e][d][s] == 1 for s in range(num_shifts))
                result[name][day] = "OFF" if available else "N/A"

    return {
        "success": True,
        "schedule": result,
        "shift_counts": shift_counts,
        "role_assignments": role_assignments,
        "employees": employees,
        "levels": levels,
        "all_roles": all_roles,
    }


def get_date_for_day(week_start_str, day_index):
    try:
        from datetime import datetime as dt, timedelta
        start = dt.strptime(week_start_str, '%Y-%m-%d')
        return (start + timedelta(days=day_index)).strftime('%d %b')
    except:
        return ''


@app.post("/generate-pdf")
def generate_pdf(req: PDFRequest):
    try:
        return _generate_pdf_inner(req)
    except Exception as e:
        import traceback
        print("PDF ERROR:", traceback.format_exc())
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=500, content={"error": str(e)})


def _generate_pdf_inner(req: PDFRequest):
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
    RED          = colors.HexColor('#A32D2D')

    SHIFT_PALETTE = [
        (AMBER_LIGHT, AMBER), (TEAL_LIGHT, TEAL), (PURPLE_LIGHT, PURPLE),
        (BLUE_LIGHT, BLUE), (PINK_LIGHT, PINK), (GREEN_LIGHT, GREEN), (CORAL_LIGHT, CORAL),
    ]

    def get_shift_color(shift_time, shift_times):
        try:
            idx = shift_times.index(shift_time)
            return SHIFT_PALETTE[idx % len(SHIFT_PALETTE)]
        except:
            return (AMBER_LIGHT, AMBER)

    def level_color(level):
        known = {'Senior': (AMBER_LIGHT, AMBER), 'Junior': (TEAL_LIGHT, TEAL), 'New Staff': (PURPLE_LIGHT, PURPLE), 'Trainee': (CORAL_LIGHT, CORAL)}
        if level in known:
            return known[level]
        palette = [(BLUE_LIGHT, BLUE), (PINK_LIGHT, PINK), (GREEN_LIGHT, GREEN), (AMBER_LIGHT, AMBER), (TEAL_LIGHT, TEAL), (PURPLE_LIGHT, PURPLE), (CORAL_LIGHT, CORAL)]
        return palette[hash(level) % len(palette)]

    all_roles = req.all_roles or []
    role_assignments = req.role_assignments or {}

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), leftMargin=14*mm, rightMargin=14*mm, topMargin=14*mm, bottomMargin=14*mm)
    story = []

    show_level_divider = req.show_level_divider if req.show_level_divider is not None else True

    week_str = req.week_start or datetime.date.today().strftime('%d %b %Y')
    if req.week_start:
        try:
            from datetime import datetime as dt, timedelta
            ws = dt.strptime(req.week_start, '%Y-%m-%d')
            we = ws + timedelta(days=6)
            week_str = f"{ws.strftime('%d %b')} – {we.strftime('%d %b %Y')}"
        except:
            pass

    # Header
    header_data = [[
        Paragraph('<b>Shiftly</b>', ParagraphStyle('logo', fontName='Helvetica-Bold', fontSize=18, textColor=AMBER)),
        Paragraph('Weekly Schedule', ParagraphStyle('rest', fontName='Helvetica-Bold', fontSize=13, textColor=TEXT_DARK)),
        Paragraph(f'Week of {week_str}<br/><font size=8 color=grey>Generated {datetime.date.today().strftime("%d %b %Y")}</font>',
            ParagraphStyle('wk', fontName='Helvetica', fontSize=10, textColor=TEXT_MID, alignment=TA_RIGHT)),
    ]]
    header_tbl = Table(header_data, colWidths=[30*mm, 180*mm, 59*mm])
    header_tbl.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LINEBELOW',(0,0),(-1,0),0.5,BORDER),('BOTTOMPADDING',(0,0),(-1,0),10),('TOPPADDING',(0,0),(-1,0),4)]))
    story.append(header_tbl)
    story.append(Spacer(1, 8))

    # Legend — shift times
    legend_items = [Paragraph('<b>Legend:</b>', ParagraphStyle('lg', fontName='Helvetica-Bold', fontSize=8, textColor=TEXT_MID))]
    legend_widths = [18*mm]
    for i, st in enumerate(req.shift_times):
        bg, fg = SHIFT_PALETTE[i % len(SHIFT_PALETTE)]
        legend_items.append(Paragraph(f'{st}', ParagraphStyle(f'lg{i}', fontName='Helvetica-Bold', fontSize=8, textColor=fg)))
        legend_widths.append(22*mm)
    legend_items.append(Paragraph('OFF = not scheduled', ParagraphStyle('lgoff', fontName='Helvetica', fontSize=8, textColor=TEXT_LIGHT)))
    legend_widths.append(40*mm)
    legend_items.append(Paragraph('N/A = not available', ParagraphStyle('lgna', fontName='Helvetica', fontSize=8, textColor=RED)))
    legend_widths.append(35*mm)
    legend_items.append(Paragraph('— = closed', ParagraphStyle('lgcl', fontName='Helvetica', fontSize=8, textColor=TEXT_LIGHT)))
    legend_widths.append(22*mm)

    # Role abbreviations in legend
    if all_roles:
        role_abbrevs = [f"{get_abbrev(r, all_roles)}={r}" for r in all_roles]
        legend_items.append(Paragraph('  |  ' + '  '.join(role_abbrevs), ParagraphStyle('lgroles', fontName='Helvetica', fontSize=8, textColor=TEXT_MID)))
        legend_widths.append(60*mm)

    page_w = landscape(A4)[0] - 28*mm
    total_used = sum(legend_widths)
    if total_used < page_w:
        legend_widths.append(page_w - total_used)
        legend_items.append(Paragraph('', ParagraphStyle('lgx', fontName='Helvetica', fontSize=8)))

    legend_tbl = Table([legend_items], colWidths=legend_widths)
    legend_tbl.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
    story.append(legend_tbl)
    story.append(Spacer(1, 4))

    # Schedule table
    col_w = 19*mm
    name_w = 52*mm
    col_widths = [name_w] + [col_w] * len(req.days) + [14*mm]

    dates_row = ['']
    days_row = ['Name']
    for di, day in enumerate(req.days):
        date_str = get_date_for_day(req.week_start, di) if req.week_start else ''
        dates_row.append(date_str)
        days_row.append(day[:3])
    dates_row.append('')
    days_row.append('Total')

    sched_data = [dates_row, days_row]
    row_styles = []
    r = 2

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
            divider = [level.upper()] + [''] * (len(req.days) + 1)
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
            row = [name]
            for day in req.days:
                val = req.schedule.get(name, {}).get(day, 'N/A')
                role = role_assignments.get(name, {}).get(day, '')
                if val not in ('—', 'N/A', 'OFF') and role and all_roles:
                    abbrev = get_abbrev(role, all_roles)
                    cell_val = f"{val} · {abbrev}"
                else:
                    cell_val = val
                row.append(cell_val)
            row.append(str(count))
            sched_data.append(row)

            row_bg = ROW_WHITE if r % 2 == 0 else ROW_ALT
            row_styles += [('BACKGROUND',(0,r),(-1,r),row_bg),('FONTNAME',(0,r),(0,r),'Helvetica-Bold'),('FONTSIZE',(0,r),(0,r),9)]

            for di, day in enumerate(req.days):
                col = di + 1
                val = req.schedule.get(name, {}).get(day, 'N/A')
                if val == '—' or day in req.closed_days:
                    row_styles += [('BACKGROUND',(col,r),(col,r),ROW_WHITE if r%2==0 else ROW_ALT),('TEXTCOLOR',(col,r),(col,r),colors.HexColor('#cccccc'))]
                elif val == 'N/A':
                    row_styles.append(('TEXTCOLOR',(col,r),(col,r),RED))
                elif val == 'OFF':
                    row_styles.append(('TEXTCOLOR',(col,r),(col,r),TEXT_LIGHT))
                else:
                    sc = get_shift_color(val, req.shift_times)
                    row_styles += [('BACKGROUND',(col,r),(col,r),sc[0]),('TEXTCOLOR',(col,r),(col,r),sc[1]),('FONTNAME',(col,r),(col,r),'Helvetica-Bold')]
            r += 1

    for di, day in enumerate(req.days):
        if day in req.closed_days:
            col = di + 1
            row_styles += [('BACKGROUND',(col,1),(col,1),BG_DARK),('TEXTCOLOR',(col,1),(col,1),AMBER)]

    base_style = TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#1e1c19')),('TEXTCOLOR',(0,0),(-1,0),colors.HexColor('#888480')),
        ('FONTNAME',(0,0),(-1,0),'Helvetica'),('FONTSIZE',(0,0),(-1,0),7.5),('ALIGN',(0,0),(-1,0),'CENTER'),
        ('TOPPADDING',(0,0),(-1,0),4),('BOTTOMPADDING',(0,0),(-1,0),4),
        ('BACKGROUND',(0,1),(-1,1),BG_DARK),('TEXTCOLOR',(0,1),(-1,1),AMBER),
        ('FONTNAME',(0,1),(-1,1),'Helvetica-Bold'),('FONTSIZE',(0,1),(-1,1),8),('ALIGN',(0,1),(-1,1),'CENTER'),
        ('TOPPADDING',(0,1),(-1,1),6),('BOTTOMPADDING',(0,1),(-1,1),6),('TEXTCOLOR',(0,1),(0,1),colors.HexColor('#c8c4bc')),
        ('ALIGN',(0,2),(0,-1),'LEFT'),('ALIGN',(1,0),(-1,-1),'CENTER'),
        ('FONTNAME',(0,2),(-1,-1),'Helvetica'),('FONTSIZE',(0,2),(-1,-1),8),
        ('TEXTCOLOR',(0,2),(0,-1),TEXT_DARK),('TEXTCOLOR',(1,2),(-1,-1),TEXT_MID),
        ('TOPPADDING',(0,2),(-1,-1),5),('BOTTOMPADDING',(0,2),(-1,-1),5),
        ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
        ('GRID',(0,0),(-1,-1),0.3,BORDER),('LINEBELOW',(0,1),(-1,1),0.5,colors.HexColor('#555')),
    ] + row_styles)

    sched_table = Table(sched_data, colWidths=col_widths, style=base_style, repeatRows=2)
    story.append(sched_table)
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