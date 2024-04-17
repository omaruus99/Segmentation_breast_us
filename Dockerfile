
# Démarrer la construction de l'image
FROM python:3.9

# Crééer un répertoire de travail dans le conteneur
WORKDIR /workspace

# Copier le fichier requirements
COPY requirements.txt .

# Mettre à jour pip
RUN pip install --upgrade pip

#Installer les dépendances de requirements
RUN pip install -r requirements.txt

#Installer les packages nécessaires à opencv (librairie de computer vision)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6

RUN pip install ipykernel -U --user --force-reinstall 


