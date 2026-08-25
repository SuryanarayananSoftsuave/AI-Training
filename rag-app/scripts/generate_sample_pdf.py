"""Generates a small multi-page test PDF with headings, a table, and a few
unusual codes/model numbers — specifically so the hybrid-vs-semantic
retrieval toggle has something concrete to show a visible difference on.
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "sample_data" / "acme_robotics_handbook.pdf"


def build() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(OUTPUT_PATH), pagesize=LETTER)
    story = []

    story.append(Paragraph("Acme Robotics — Employee Handbook", styles["Title"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("1. Leave Policy", styles["Heading1"]))
    story.append(
        Paragraph(
            "Acme Robotics grants every full-time employee a combined annual leave allowance "
            "split across three categories: casual, sick, and earned leave. Leave requests must "
            "be submitted through the HR portal at least three business days in advance, except "
            "for sick leave, which may be reported on the day of absence. Unused earned leave "
            "carries over to the next calendar year up to a maximum of ten days; casual and sick "
            "leave do not carry over.",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 10))

    story.append(Paragraph("1.1 Leave Type Reference", styles["Heading2"]))
    table_data = [
        ["Code", "Leave Type", "Annual Entitlement", "Carries Over"],
        ["PL-01", "Casual Leave", "12 days", "No"],
        ["PL-02", "Sick Leave", "10 days", "No"],
        ["PL-03", "Earned Leave", "18 days", "Yes, up to 10 days"],
        ["PL-04", "Parental Leave", "26 weeks", "No"],
    ]
    table = Table(table_data, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0E6D62")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(table)
    story.append(PageBreak())

    story.append(Paragraph("2. Expense Reimbursement", styles["Heading1"]))
    story.append(
        Paragraph(
            "Employees may claim reimbursement for pre-approved business expenses by submitting "
            "a claim referencing the internal cost center code and attaching original receipts. "
            "Claims must be filed within 30 days of the expense date. Reimbursements above $500 "
            "require a second approval from the department head.",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 10))

    story.append(Paragraph("2.1 Equipment Warranty Claims", styles["Heading2"]))
    story.append(
        Paragraph(
            "Field service technicians replacing a defective actuator unit must reference the "
            "warranty claim prefix WARR-4471 on the replacement form. The affected hardware "
            "model, the RX-4471-B servo controller, carries a manufacturer warranty of 36 months "
            "from the date of installation. Claims filed without the WARR-4471 prefix will be "
            "rejected by the parts department.",
            styles["BodyText"],
        )
    )
    story.append(PageBreak())

    story.append(Paragraph("3. Remote Work Policy", styles["Heading1"]))
    story.append(
        Paragraph(
            "Employees in engineering and support roles may work remotely up to three days per "
            "week, subject to manager approval. Remote work requests are submitted via the same "
            "HR portal used for leave requests. Employees working remotely must remain reachable "
            "during core hours, 10:00 AM to 4:00 PM local time.",
            styles["BodyText"],
        )
    )

    doc.build(story)
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    build()
