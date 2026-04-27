from __future__ import annotations
import random
import uuid
from dataclasses import dataclass
from typing import Callable, Literal


@dataclass
class Task:
    task_id: str
    task_description: str
    ground_truth: str
    correctness_mode: Literal["exact", "fuzzy", "semantic"]
    domain: str


@dataclass
class TaskTemplate:
    template_id: str
    question_template: str
    # slot_name -> list of possible fill values
    slots: dict[str, list[str]]
    # "|".join(sorted fill values) -> correct answer
    ground_truth_map: dict[str, str]
    correctness_mode: Literal["exact", "fuzzy", "semantic"]
    domain: str


TASK_TEMPLATES: list[TaskTemplate] = [
    # ------------------------------------------------------------------ Medical
    # Drug and condition are correlated — sampled as a pair to avoid invalid combos.
    TaskTemplate(
        template_id="med_dosage_adult",
        question_template="What is the standard adult oral dosage of {drug_condition}?",
        slots={
            "drug_condition": [
                "ibuprofen for pain relief",
                "amoxicillin for bacterial infection",
                "metformin for type 2 diabetes",
                "lisinopril for hypertension",
                "atorvastatin for high cholesterol",
            ],
        },
        ground_truth_map={
            "ibuprofen for pain relief": "200-400 mg every 4-6 hours; OTC maximum 1200 mg per day",
            "amoxicillin for bacterial infection": "250-500 mg every 8 hours, or 500-875 mg every 12 hours",
            "metformin for type 2 diabetes": "500 mg twice daily with meals; titrate to 2000 mg per day",
            "lisinopril for hypertension": "10 mg once daily; usual maintenance 20-40 mg per day",
            "atorvastatin for high cholesterol": "10-20 mg once daily; maximum 80 mg per day",
        },
        correctness_mode="fuzzy",
        domain="medical",
    ),
    TaskTemplate(
        template_id="med_contraindication",
        question_template="What is a major contraindication for {drug}?",
        slots={
            "drug": ["warfarin", "methotrexate", "lithium", "digoxin", "phenytoin"],
        },
        ground_truth_map={
            "warfarin": "active significant bleeding or hemorrhagic stroke",
            "methotrexate": "pregnancy and significant renal impairment",
            "lithium": "severe renal disease and dehydration",
            "digoxin": "ventricular fibrillation and hypertrophic obstructive cardiomyopathy",
            "phenytoin": "concurrent use of delavirdine and sinus bradycardia",
        },
        correctness_mode="fuzzy",
        domain="medical",
    ),
    # ---------------------------------------------------------------- Financial
    TaskTemplate(
        template_id="fin_revenue",
        question_template="What was {company}'s reported annual net revenue for fiscal year {year}?",
        slots={
            "company": ["Apple", "Microsoft", "Google", "Amazon", "Meta"],
            "year": ["2022", "2023"],
        },
        ground_truth_map={
            "2022|Apple": "$394.3 billion",
            "2023|Apple": "$383.3 billion",
            "2022|Microsoft": "$198.3 billion",
            "2023|Microsoft": "$211.9 billion",
            "2022|Google": "$282.8 billion",
            "2023|Google": "$307.4 billion",
            "2022|Amazon": "$513.98 billion",
            "2023|Amazon": "$574.8 billion",
            "2022|Meta": "$116.6 billion",
            "2023|Meta": "$134.9 billion",
        },
        correctness_mode="fuzzy",
        domain="financial",
    ),
    TaskTemplate(
        template_id="fin_founding_year",
        question_template="In what year was {company} founded?",
        slots={
            "company": ["Apple", "Microsoft", "Amazon", "Tesla", "Nvidia"],
        },
        ground_truth_map={
            "Apple": "1976",
            "Microsoft": "1975",
            "Amazon": "1994",
            "Tesla": "2003",
            "Nvidia": "1993",
        },
        correctness_mode="exact",
        domain="financial",
    ),
    # --------------------------------------------------------------- Historical
    TaskTemplate(
        template_id="hist_year",
        question_template="In what year did {event} occur?",
        slots={
            "event": [
                "the Berlin Wall fall",
                "the first moon landing",
                "World War II end",
                "the French Revolution begin",
                "the signing of the Magna Carta",
            ],
        },
        ground_truth_map={
            "the Berlin Wall fall": "1989",
            "the first moon landing": "1969",
            "World War II end": "1945",
            "the French Revolution begin": "1789",
            "the signing of the Magna Carta": "1215",
        },
        correctness_mode="exact",
        domain="historical",
    ),
    TaskTemplate(
        template_id="hist_leader",
        question_template="Who was the {country} head of government during {event}?",
        slots={
            "country": ["United States", "United Kingdom"],
            "event": ["World War II", "the moon landing"],
        },
        ground_truth_map={
            "World War II|United States": "Franklin D. Roosevelt (and Harry S. Truman from 1945)",
            "World War II|United Kingdom": "Winston Churchill (and Clement Attlee from July 1945)",
            "the moon landing|United States": "Richard Nixon",
            "the moon landing|United Kingdom": "Harold Wilson",
        },
        correctness_mode="fuzzy",
        domain="historical",
    ),
    # ------------------------------------------------------------ Scientific
    TaskTemplate(
        template_id="sci_constant",
        question_template="What is the value of {constant}?",
        slots={
            "constant": [
                "the speed of light in a vacuum",
                "Avogadro's number",
                "the gravitational constant G",
                "Planck's constant",
                "Boltzmann's constant",
            ],
        },
        ground_truth_map={
            "the speed of light in a vacuum": "299,792,458 metres per second",
            "Avogadro's number": "6.022 x 10^23 per mole",
            "the gravitational constant G": "6.674 x 10^-11 N m^2 kg^-2",
            "Planck's constant": "6.626 x 10^-34 joule-seconds",
            "Boltzmann's constant": "1.381 x 10^-23 joules per kelvin",
        },
        correctness_mode="fuzzy",
        domain="scientific",
    ),
    TaskTemplate(
        template_id="sci_element",
        question_template="What is the atomic number of {element}?",
        slots={
            "element": ["carbon", "oxygen", "gold", "iron", "uranium"],
        },
        ground_truth_map={
            "carbon": "6",
            "oxygen": "8",
            "gold": "79",
            "iron": "26",
            "uranium": "92",
        },
        correctness_mode="exact",
        domain="scientific",
    ),
    # --------------------------------------------------------------- Legal
    TaskTemplate(
        template_id="legal_sol",
        question_template=(
            "Under {jurisdiction} law, what is the statute of limitations for {claim_type}?"
        ),
        slots={
            "jurisdiction": ["California", "New York", "Texas", "UK", "US federal"],
            "claim_type": ["breach of written contract", "personal injury", "fraud"],
        },
        ground_truth_map={
            "breach of written contract|California": "4 years under California Code of Civil Procedure § 337",
            "breach of written contract|New York": "6 years under CPLR § 213",
            "breach of written contract|Texas": "4 years under Texas Civil Practice & Remedies Code § 16.004",
            "breach of written contract|UK": "6 years under Limitation Act 1980 s.5",
            "breach of written contract|US federal": "varies; 6 years for most federal contracts",
            "fraud|California": "3 years from discovery under CCP § 338",
            "fraud|New York": "6 years, or 2 years from discovery if later, under CPLR § 213(8)",
            "fraud|Texas": "4 years under Texas Civil Practice & Remedies Code § 16.004",
            "fraud|UK": "6 years from discovery under Limitation Act 1980 s.32",
            "fraud|US federal": "5 years under 18 U.S.C. § 3282 (criminal)",
            "personal injury|California": "2 years under California Code of Civil Procedure § 335.1",
            "personal injury|New York": "3 years under CPLR § 214",
            "personal injury|Texas": "2 years under Texas Civil Practice & Remedies Code § 16.003",
            "personal injury|UK": "3 years under Limitation Act 1980 s.11",
            "personal injury|US federal": "varies by tort; FTCA requires 2-year administrative claim",
        },
        correctness_mode="fuzzy",
        domain="legal",
    ),
]


class TaskBank:
    def __init__(
        self,
        templates: list[TaskTemplate] = TASK_TEMPLATES,
        seed: int | None = None,
    ):
        self._templates = templates
        self._rng = random.Random(seed)

    def sample(self) -> Task:
        """Sample one task procedurally by slot-filling a random template."""
        template = self._rng.choice(self._templates)

        filled: dict[str, str] = {
            slot_name: self._rng.choice(options)
            for slot_name, options in template.slots.items()
        }

        question = template.question_template.format(**filled)

        # Build lookup key: sorted slot values joined by "|"
        key = "|".join(filled[k] for k in sorted(filled.keys()))
        ground_truth = template.ground_truth_map.get(key)

        if ground_truth is None:
            # Fallback: find any key whose parts are all in filled values
            fv_set = set(filled.values())
            for k, v in template.ground_truth_map.items():
                if fv_set.issuperset(k.split("|")):
                    ground_truth = v
                    break

        if ground_truth is None:
            ground_truth = "Information not available"

        return Task(
            task_id=str(uuid.uuid4()),
            task_description=question,
            ground_truth=ground_truth,
            correctness_mode=template.correctness_mode,
            domain=template.domain,
        )
