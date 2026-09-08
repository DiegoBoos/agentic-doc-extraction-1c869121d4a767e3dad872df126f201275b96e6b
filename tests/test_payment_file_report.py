from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from pypdf import PdfWriter

from app.main import app

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "payment_file_reports"


def _read_fixture(name: str) -> bytes:
    return (FIXTURES_DIR / name).read_bytes()


def test_parse_payment_file_single_beneficiary() -> None:
    pdf_bytes = _read_fixture("single_line_beneficiary.pdf")

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/payment-files/parse",
            files={"file": ("informe.pdf", BytesIO(pdf_bytes), "application/pdf")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["clave_empresa"] == "MALLAMAS"
    assert body["nombre_fichero"] == "ES356-2026"
    assert body["fecha_proceso"] == "2026-03-11"
    assert body["fecha_generacion_informe"] == "2026-03-18T08:47:34"
    assert body["ordenes"] == 1
    assert body["importe_total"] == "1275110.30"

    assert len(body["beneficiarios"]) == 1
    beneficiary = body["beneficiarios"][0]
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
    assert len(body["beneficiarios"]) == 1
    beneficiary = body["beneficiarios"][0]
    # Name wraps across two lines in the source PDF; the parser must join them.
    assert beneficiary["nombre"] == "ESE HOSPITAL SAN GABRIEL ARCANGEL"
    assert beneficiary["tipo_cuenta"] is None


def test_parse_payment_file_rejects_non_pdf_extension() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/payment-files/parse",
            files={"file": ("informe.txt", BytesIO(b"not a pdf"), "text/plain")},
        )

    assert response.status_code == 400


def test_parse_payment_file_rejects_unrecognized_layout() -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = BytesIO()
    writer.write(buffer)
    buffer.seek(0)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/payment-files/parse",
            files={"file": ("informe.pdf", buffer, "application/pdf")},
        )

    assert response.status_code == 422
    assert response.json()["detail"]
