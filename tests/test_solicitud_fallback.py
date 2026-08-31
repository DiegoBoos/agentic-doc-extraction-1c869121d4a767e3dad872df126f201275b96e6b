from app.schemas.authorization import AuthorizationExtraction, AuthorizationResponse
from app.services.processing_pipeline import normalize_authorizations


def test_authorization_response_keeps_solicitud_only_records() -> None:
    response = AuthorizationResponse(
        authorizations=[AuthorizationExtraction(numero_solicitud="4608")]
    )

    assert len(response.authorizations) == 1
    assert response.authorizations[0].numero_autorizacion is None
    assert response.authorizations[0].numero_solicitud == "4608"


def test_normalize_authorizations_builds_fallback_entries_from_solicitud_numbers() -> None:
    markdown = """
    SOLICITUD DE AUTORIZACIÓN DE SERVICIOS Y TECNOLOGÍAS EN SALUD
    2607184608 No. Solicitud 4608 Fecha y hora : 18/07/2026 1:16:40 p. m.
    ...
    2607184609 No. Solicitud 4609 Fecha y hora : 18/07/2026 1:21:34 p. m.
    """

    payload = normalize_authorizations({"authorizations": []}, markdown=markdown)

    assert payload["authorizations"] == [
        {
            "numero_autorizacion": None,
            "numero_solicitud": "4608",
            "fecha_autorizacion": None,
            "prestador_autorizado": {},
            "datos_paciente": {},
            "servicios_autorizados": {
                "ubicacion_paciente": None,
                "grupo_servicio": None,
                "items": [],
            },
            "vigencia": None,
        },
        {
            "numero_autorizacion": None,
            "numero_solicitud": "4609",
            "fecha_autorizacion": None,
            "prestador_autorizado": {},
            "datos_paciente": {},
            "servicios_autorizados": {
                "ubicacion_paciente": None,
                "grupo_servicio": None,
                "items": [],
            },
            "vigencia": None,
        },
    ]
