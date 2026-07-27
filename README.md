# Kivo — Asistente virtual con RAG

Agente conversacional que responde preguntas de soporte al cliente de **Kivo**, una tienda en
línea ficticia de tecnología y accesorios, usando como única fuente de verdad los documentos
oficiales de la tienda (políticas de envío y devoluciones, preguntas frecuentes, términos y
condiciones, y el catálogo de productos).

El objetivo del proyecto es mostrar un flujo completo de **RAG (Retrieval-Augmented
Generation)**: lectura de documentos reales (PDF/CSV), indexado por embeddings, recuperación
semántica y generación de respuestas ancladas al contenido, todo empaquetado y desplegado como
una aplicación funcional.

## Cómo funciona

1. **Ingesta** (`ingest.py`): lee los PDF y el CSV de `data/`, los trocea en fragmentos y genera
   un embedding para cada uno con el modelo `gemini-embedding-2`. El resultado se guarda como un
   índice vectorial simple (un array de NumPy + metadatos en JSON) en `index/`.
2. **Recuperación**: cuando llega una pregunta, se genera su embedding, se compara por similitud
   coseno contra el índice y se seleccionan los fragmentos más relevantes.
3. **Generación**: esos fragmentos se arman como contexto y se envían junto con la pregunta al
   modelo `gemini-3.6-flash` (vía la Interactions API de Gemini), con una instrucción de sistema
   que obliga al modelo a responder solo con base en ese contexto y a admitir cuando no tiene la
   información.
4. **Interfaz**: una app de Streamlit con historial de conversación y las fuentes usadas en cada
   respuesta.

```
data/*.pdf, data/*.csv
        │
        ▼
   ingest.py  ──►  embeddings (Gemini)  ──►  index/ (NumPy + JSON)
                                                   │
usuario ──► app.py (Streamlit) ──► embedding de la pregunta ──► búsqueda por similitud
                                                   │
                                          contexto recuperado
                                                   │
                                    gemini-3.6-flash (Interactions API)
                                                   │
                                              respuesta final
```

No se usa ningún framework de orquestación (LangChain, LlamaIndex, etc.): la recuperación y el
armado del prompt están escritos directamente para mantener el flujo simple y fácil de auditar.

## Tecnologías

- Python 3.12
- [google-genai](https://github.com/googleapis/python-genai) — SDK oficial de la API de Gemini
  (embeddings con `gemini-embedding-2`, generación con `gemini-3.6-flash`)
- Streamlit — interfaz de chat
- pypdf / pandas — lectura de los documentos fuente
- NumPy — índice vectorial y similitud coseno
- Docker / Docker Compose — empaquetado y despliegue

## Documentos que conoce el agente

| Archivo | Contenido |
|---|---|
| `data/politica_envios_devoluciones.pdf` | Tiempos y costos de envío, devoluciones, reembolsos, garantía |
| `data/preguntas_frecuentes.pdf` | FAQ de pedidos, pagos, envíos, productos y soporte |
| `data/terminos_y_condiciones.pdf` | Términos de uso, privacidad, propiedad intelectual |
| `data/catalogo_productos.csv` | 15 productos con precio, stock, categoría y garantía |

## Cómo ejecutarlo

### 1. Conseguir una API key de Gemini (gratis)

Crear una en <https://aistudio.google.com/apikey> (no pide tarjeta de crédito).

### 2. Local, sin Docker

```bash
git clone https://github.com/luigivivas/kivo-ai-agent.git
cd kivo-ai-agent
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
cp .env.example .env          # y pegar tu GEMINI_API_KEY ahí

python ingest.py              # construye el índice (solo la primera vez)
streamlit run app.py
```

Abrir <http://localhost:8501>.

### 3. Con Docker

```bash
cp .env.example .env          # y pegar tu GEMINI_API_KEY ahí
docker compose up --build
```

Abrir <http://localhost:8600>. El contenedor construye el índice automáticamente al arrancar si
todavía no existe.

## Ejemplos de preguntas

- ¿Cuánto cuesta el smartwatch Kivo Fit 2 y qué garantía tiene?
- Compré unos audífonos y no me gustaron, ¿los puedo devolver?
- ¿A qué países hacen envíos y cuánto tardan?
- ¿Qué pasa si mi pedido llega dañado?
- ¿Puedo pagar en cuotas?
- ¿Qué hacen con mis datos personales?

## Ejemplos de respuestas

Transcripciones reales generadas por el agente ya desplegado (no son texto de ejemplo escrito a
mano):

> **¿Cuánto cuesta el smartwatch Kivo Fit 2 y qué garantía tiene?**
> El Smartwatch Kivo Fit 2 tiene un precio de $89.90 USD y cuenta con una garantía de 12 meses
> por defectos de fábrica. Además, este modelo incluye envío gratis e integra GPS, monitor de
> ritmo cardíaco y una batería con duración de hasta 7 días.

> **Compré unos audífonos y no me gustaron, ¿los puedo devolver?**
> Puedes devolverlos dentro de los 30 días calendario posteriores a la entrega, siempre que el
> producto esté en su empaque original y con todos sus accesorios. Ten en cuenta que, por motivos
> de higiene, los audífonos de tipo *in-ear* ya usados solo se aceptan para devolución si
> presentan una falla de fábrica comprobable. Como la devolución es por cambio de opinión, el
> costo del envío de retorno corre por tu cuenta, y puedes solicitar el trámite desde la sección
> "Mis pedidos" o escribiendo a soporte@kivo-tienda.com.

> **¿A qué países hacen envíos y cuánto tardan?**
> Realizamos envíos a Colombia, México, Perú y Chile. El tiempo de entrega oscila entre 2 y 10
> días hábiles según la ubicación: toma de 2 a 4 días hábiles para ciudades principales, de 4 a 6
> días hábiles para ciudades intermedias, y de 6 a 10 días hábiles para zonas rurales o de difícil
> acceso.

> **¿Cuál es la capital de Francia?** _(pregunta fuera de contexto, a propósito)_
> No tengo esa información en mis documentos. Si tienes alguna consulta relacionada con nuestros
> productos, compras o políticas en Kivo, con gusto te ayudo, o puedes escribirnos directamente a
> soporte@kivo-tienda.com.

## Despliegue

La aplicación corre en un contenedor Docker sobre una VPS propia, aislada del resto de servicios
que corren en esa máquina (contenedor, puerto y volumen dedicados, sin dependencias compartidas).

- **URL pública:** http://31.97.132.21:8600

## Estructura del proyecto

```
kivo-ai-agent/
├── app.py                # interfaz Streamlit
├── ingest.py             # construye el índice vectorial
├── src/
│   ├── loaders.py        # lectura y troceado de PDF/CSV
│   ├── gemini_client.py  # embeddings + generación (SDK de Gemini)
│   ├── store.py          # índice vectorial (NumPy) + búsqueda por similitud
│   └── agent.py          # arma el contexto y orquesta la respuesta
├── data/                 # documentos fuente (PDF/CSV)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```
