from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from pypdf import PdfWriter

from app.main import app
from app.services.document_parser import ParsedTablesResult

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "payment_file_reports"


def _read_fixture(name: str) -> bytes:
    return (FIXTURES_DIR / name).read_bytes()


def _build_minimal_pdf(pages: list[list[str]]) -> bytes:
    """Build a minimal multi-page PDF with the given lines of text per page.

    Used to synthesize fixtures for parsing tests without depending on
    external files or PDF-generation libraries.
    """

    def esc(s: str) -> str:
        return s.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")

    objects: list[bytes] = []
    objects.append(b"")  # placeholder for catalog (object 1)
    objects.append(b"")  # placeholder for pages tree (object 2)

    page_object_numbers = []
    font_object_number = 3 + len(pages) * 2  # after page + content-stream objects
    next_obj = 3

    page_bodies = []
    for lines in pages:
        page_obj_num = next_obj
        content_obj_num = next_obj + 1
        next_obj += 2
        page_object_numbers.append(page_obj_num)

        content_lines = []
        y = 780
        for line in lines:
            content_lines.append(f"BT /F1 10 Tf 40 {y} Td ({esc(line)}) Tj ET")
            y -= 14
        content_stream = "\n".join(content_lines).encode("cp1252", errors="replace")

        page_bodies.append(
            (
                page_obj_num,
                (
                    b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                    b"/Resources << /Font << /F1 "
                    + str(font_object_number).encode()
                    + b" 0 R >> >> "
                    b"/Contents " + str(content_obj_num).encode() + b" 0 R >>"
                ),
            )
        )
        page_bodies.append(
            (
                content_obj_num,
                b"<< /Length "
                + str(len(content_stream)).encode()
                + b" >>\nstream\n"
                + content_stream
                + b"\nendstream",
            )
        )

    kids = " ".join(f"{n} 0 R" for n in page_object_numbers)
    objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_object_numbers)} >>".encode()
    objects.extend(body for _, body in page_bodies)
    objects.append(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
    )

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"

    xref_offset = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF"
    ).encode()
    return bytes(out)


def _report_lines(
    *,
    timestamp: str,
    clave_empresa: str,
    nombre_fichero: str,
    beneficiario: str,
    numero_identificacion: str,
    cuenta: str,
    importe: str,
) -> list[str]:
    return [
        "Informe Detallado del Fichero",
        timestamp,
        "Informacion del fichero",
        f"Clave Empresa: {clave_empresa} Cuenta de Cargo: CC - 00130445000100010416",
        f"F.de proceso: 11-03-2026 Nombre Fichero: {nombre_fichero}",
        "Referencia: DVP 12/03/2026 Órdenes: 1",
        f"Producto: Pagos a Proveedores Importe Total: {importe}",
        "BENEFICIARIO CUENTA BENEFICIARIA IMPORTE (COP) MOTIVO",
        f"{beneficiario} {cuenta} {importe} AUTORIZADO",
        f"Tipo de identificación: NIT Persona Jurídica Nº identificación: {numero_identificacion}",
        f"Nombre: {beneficiario} E-mail:",
        "Dirección 1: ARMENIA Dirección 2:",
        "Forma de Pago: Abono/Cargo cuenta Tipo de cuenta: Cuenta Corriente",
        f"Banco: 0051 - BANCO DAVIVIENDA Cuenta-Tarjeta: {cuenta}",
        "Código Oficina Pagadora: 0 Fecha Límite Vencimiento:",
        f"Importe: {importe} Motivo Devolución: AUTORIZADO",
        "Concepto 1: 52 SUBSIDIADO Concepto 2:",
    ]


def test_parse_payment_file_single_beneficiary() -> None:
    pdf_bytes = _read_fixture("single_line_beneficiary.pdf")

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/payment-files/parse",
            files={"file": ("informe.pdf", BytesIO(pdf_bytes), "application/pdf")},
        )

    assert response.status_code == 200
    body = response.json()
    assert len(body["reports"]) == 1
    report = body["reports"][0]
    assert report["clave_empresa"] == "MALLAMAS"
    assert report["nombre_fichero"] == "ES356-2026"
    assert report["fecha_proceso"] == "2026-03-11"
    assert report["fecha_generacion_informe"] == "2026-03-18T08:47:34"
    assert report["ordenes"] == 1
    assert report["importe_total"] == "1275110.30"

    assert len(report["beneficiarios"]) == 1
    beneficiary = report["beneficiarios"][0]
    assert beneficiary["nombre"] == "CLINICA SAGRADA FAMILIA"
    assert beneficiary["numero_identificacion"] == "0000009013523533"
    assert beneficiary["banco"] == "0051 - BANCO DAVIVIENDA"
    assert beneficiary["cuenta_tarjeta"] == "076069995835"
    assert beneficiary["importe"] == "1275110.30"
    assert beneficiary["motivo_devolucion"] == "AUTORIZADO"
    assert beneficiary["email"] is None
    assert beneficiary["fecha_limite_vencimiento"] is None


def test_parse_payment_file_wrapped_beneficiary_name() -> None:
    pdf_bytes = _read_fixture("wrapped_beneficiary_name.pdf")

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/payment-files/parse",
            files={"file": ("informe.pdf", BytesIO(pdf_bytes), "application/pdf")},
        )

    assert response.status_code == 200
    body = response.json()
    assert len(body["reports"]) == 1
    beneficiary = body["reports"][0]["beneficiarios"][0]
    # Name wraps across two lines in the source PDF; the parser must join them.
    assert beneficiary["nombre"] == "ESE HOSPITAL SAN GABRIEL ARCANGEL"
    assert beneficiary["tipo_cuenta"] is None


def test_parse_payment_file_multiple_reports_in_one_pdf() -> None:
    pdf_bytes = _build_minimal_pdf(
        [
            _report_lines(
                timestamp="18/Mar/26 08:47:34",
                clave_empresa="MALLAMAS",
                nombre_fichero="ES356-2026",
                beneficiario="CLINICA SAGRADA FAMILIA",
                numero_identificacion="0000009013523533",
                cuenta="076069995835",
                importe="1.275.110,30",
            ),
            _report_lines(
                timestamp="26/Mar/26 07:55:44",
                clave_empresa="MALLAMAS",
                nombre_fichero="ES363-2026",
                beneficiario="DIAGNOSTIMED S A",
                numero_identificacion="0000008305142409",
                cuenta="71651605984",
                importe="23.439.493,00",
            ),
        ]
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/payment-files/parse",
            files={"file": ("informe.pdf", BytesIO(pdf_bytes), "application/pdf")},
        )

    assert response.status_code == 200
    body = response.json()
    assert len(body["reports"]) == 2
    assert body["reports"][0]["nombre_fichero"] == "ES356-2026"
    assert body["reports"][0]["beneficiarios"][0]["nombre"] == "CLINICA SAGRADA FAMILIA"
    assert body["reports"][1]["nombre_fichero"] == "ES363-2026"
    assert body["reports"][1]["beneficiarios"][0]["nombre"] == "DIAGNOSTIMED S A"


def _real_scanned_report_tables() -> ParsedTablesResult:
    """Table grid captured verbatim from a real Azure Document Intelligence
    prebuilt-layout call against tests/fixtures/payment_file_reports/scanned_report.pdf
    (an image-only PDF built from an actual BBVA report screenshot). Used to
    keep the OCR-fallback tests offline/deterministic while still exercising
    the exact shape Azure returns for this document.
    """
    header_table = [
        ["Clave Empresa:", "MALLAMAS", "Cuenta de Cargo:", "CC - 00130445000100010416"],
        ["F.de proceso:", "11-03-2026", "Nombre Fichero:", "ES356-2026"],
        ["Referencia:", "DVP 12/03/2026", "Órdenes:", "1"],
        ["Producto:", "Pagos a Proveedores", "Importe Total:", "1.275.110,30"],
    ]
    beneficiary_table = [
        ["BENEFICIARIO", "CUENTA BENEFICIARIA", "IMPORTE (COP)", "MOTIVO"],
        ["CLINICA SAGRADA FAMILIA", "076069995835", "1.275.110,30", "AUTORIZADO"],
        ["", "", "", ""],
        [
            "Tipo de identificación:",
            "NIT Persona Jurídica",
            "Nº identificación:",
            "0000009013523533",
        ],
        ["Nombre:", "CLINICA SAGRADA FAMILIA", "E-mail:", ""],
        ["Dirección 1:", "ARMENIA", "Dirección 2:", ""],
        ["Forma de Pago:", "Abono/Cargo cuenta", "Tipo de cuenta:", "Cuenta Corriente"],
        ["Banco:", "0051 - BANCO DAVIVIENDA", "Cuenta-Tarjeta:", "076069995835"],
        ["Código Oficina Pagadora:", "0", "Fecha Límite Vencimiento:", ""],
        ["Importe:", "1.275.110,30", "Motivo Devolución:", "AUTORIZADO"],
        ["Concepto 1:", "52 SUBSIDIADO", "Concepto 2:", ""],
    ]
    content = (
        "BBVA Net Cash\nInforme Detallado del Fichero\n18/Mar/26 08:47:34\nInformación del fichero"
    )
    return ParsedTablesResult(content=content, tables=[header_table, beneficiary_table])


def test_parse_payment_file_falls_back_to_ocr_when_no_text_layer() -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    buffer = BytesIO()
    writer.write(buffer)
    scanned_pdf_bytes = buffer.getvalue()

    with TestClient(app) as client:
        calls: list[dict] = []

        def fake_parse_tables(**kwargs):
            calls.append(kwargs)
            return _real_scanned_report_tables()

        client.app.state.parser.parse_tables = fake_parse_tables

        response = client.post(
            "/api/v1/payment-files/parse",
            files={"file": ("escaneado.pdf", BytesIO(scanned_pdf_bytes), "application/pdf")},
        )

    assert response.status_code == 200
    assert len(calls) == 1
    body = response.json()
    assert len(body["reports"]) == 1
    report = body["reports"][0]
    assert report["clave_empresa"] == "MALLAMAS"
    assert report["nombre_fichero"] == "ES356-2026"
    assert report["importe_total"] == "1275110.30"
    beneficiary = report["beneficiarios"][0]
    assert beneficiary["nombre"] == "CLINICA SAGRADA FAMILIA"
    assert beneficiary["banco"] == "0051 - BANCO DAVIVIENDA"
    assert beneficiary["importe"] == "1275110.30"


def test_parse_payment_file_scanned_screenshot_fixture_via_ocr() -> None:
    """End-to-end against an image-only PDF built from a real report screenshot
    (no text layer at all), with the Azure call mocked for determinism."""
    scanned_pdf_bytes = _read_fixture("scanned_report.pdf")

    with TestClient(app) as client:
        client.app.state.parser.parse_tables = lambda **_: _real_scanned_report_tables()

        response = client.post(
            "/api/v1/payment-files/parse",
            files={"file": ("captura.pdf", BytesIO(scanned_pdf_bytes), "application/pdf")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["reports"][0]["nombre_fichero"] == "ES356-2026"
    assert body["reports"][0]["beneficiarios"][0]["nombre"] == "CLINICA SAGRADA FAMILIA"


def test_parse_payment_file_reports_error_when_ocr_unavailable() -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    buffer = BytesIO()
    writer.write(buffer)
    scanned_pdf_bytes = buffer.getvalue()

    with TestClient(app) as client:

        def unavailable_parse_tables(**_kwargs):
            raise ValueError("Parser provider 'azure' is not available or not configured.")

        client.app.state.parser.parse_tables = unavailable_parse_tables

        response = client.post(
            "/api/v1/payment-files/parse",
            files={"file": ("escaneado.pdf", BytesIO(scanned_pdf_bytes), "application/pdf")},
        )

    assert response.status_code == 422


def test_parse_payment_file_rejects_non_pdf_extension() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/payment-files/parse",
            files={"file": ("informe.txt", BytesIO(b"not a pdf"), "text/plain")},
        )

    assert response.status_code == 400


def test_parse_payment_file_rejects_unrecognized_layout() -> None:
    pdf_bytes = _build_minimal_pdf([["Hello world", "This is not a payment file report."]])

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/payment-files/parse",
            files={"file": ("informe.pdf", BytesIO(pdf_bytes), "application/pdf")},
        )

    assert response.status_code == 422
    assert response.json()["detail"]
