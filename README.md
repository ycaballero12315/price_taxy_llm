# 🚀 Guía de Instalación

Guía paso a paso para ejecutar el proyecto de predicción de tarifas de taxi NYC.

---

## 📋 Requisitos Previos

Antes de comenzar, asegúrate de tener instalado:

- **Python 3.10** ([Descargar aquí](https://www.python.org/downloads/))
- **pip** (incluido con Python)
- **Git** (opcional, para clonar el repositorio)

### Verificar instalación de Python

```bash
python --version
# o en algunos sistemas
python3 --version
```

Deberías ver algo como: `Python 3.10.x`

---

## 🔧 Instalación

### Opción 1: Instalación Rápida (Recomendada)

```bash
# 1. Clonar el repositorio (o descargar ZIP)
git clone https://github.com/ycaballero12315/price_taxy_llm.git
cd price_taxy_llm

# 2. Crear entorno virtual
python -m venv .venv

# 3. Activar entorno virtual
# En Windows (CMD)
. .venv\Scripts\activate

# En Windows (PowerShell)
. .venv\Scripts\Activate.ps1

# En Linux/Mac
. .venv/bin/activate

# 4. Instalar dependencias
pip install -r requirements.txt

# 5. Ejecutar el proyecto
python main.py
```
