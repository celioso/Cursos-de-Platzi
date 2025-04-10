import json
import urllib.request

def generar(event, context):
    try:
        # Extraer el cuerpo del evento
        body = json.loads(event.get("body", "{}"))

        romantico = body["romantico"]
        numero_max_palabras = body["numero_max_palabras"]
        lenguaje = body["lenguaje_de_programacion"]

        # Construir el prompt
        tono = "romántico" if romantico else "neutral"
        prompt = (
            f"Escribe un poema {tono} en {lenguaje}, con un máximo de "
            f"{numero_max_palabras} palabras, inspirado en el mundo de los programadores."
        )

        # Preparar datos para la petición a OpenAI
        data = json.dumps({
            "model": "text-davinci-003",
            "prompt": prompt,
            "temperature": 0.7
        }).encode("utf-8")

        # Preparar la solicitud
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_KEY}",
            "OpenAI-Organization": OPENAI_ORG
        }

        req = urllib.request.Request(
            url="https://api.openai.com/v1/completions",
            data=data,
            headers=headers,
            method="POST"
        )

        # Enviar la solicitud
        with urllib.request.urlopen(req) as response:
            response_data = response.read().decode("utf-8")
            completion = json.loads(response_data)

        # Obtener el texto generado
        poema = completion["choices"][0]["text"].strip()

        return {
            "statusCode": 200,
            "body": json.dumps({"poema": poema})
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }

# Variables de entorno
import os
OPENAI_KEY = os.getenv("OPENAIKEY")
OPENAI_ORG = os.getenv("OPENAIORG")
