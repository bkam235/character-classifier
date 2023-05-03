# Character classifier für Ziffern-OCR mit niedriger Auflösung

## Trainingsdaten

Ordner `/container` enthält Code für ein Docker-Image zum Generieren der Trainingsdaten. Dazu in `/container/app/fonts` die Schriftarten als `.ttf` oder `.ttc` ablegen.

Dann mit shell in `/container` aufrufen:

`(sudo) docker build -t char_data .`

`(sudo) docker -v /app/out:/app/out:Z --name char_data -t -d -p 5100:80 char_data`

`sudo docker exec -it char_data /bin/bash`

In der interaktiven shell im Container dann ausführen:

`python3 generate_data.py`

## Ordnerstruktur

`split_data.py` legt die erforderliche Ordnerstruktur an und verschiebt die generierten Trainingsdaten.

## Training und Prediciton

`train.py` trainiert ein Mini-VGG-16 und speichert es als `numeric_char_classifier.h5`.

`predict.py` schreibt auf eine JPG-Datei (im Beispiel `IBAN.jpg`) die erkannten Ziffer an die Konturen.