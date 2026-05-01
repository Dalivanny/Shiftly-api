from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from ortools.sat.python import cp_model

app = FastAPI()

# Allow your Next.js app to call this API
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

    # Create shift variables
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

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30.0
    status = solver.solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"error": "No valid schedule found. Check availability and staffing rules."}

    # Build result
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