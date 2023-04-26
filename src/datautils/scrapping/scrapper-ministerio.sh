#!/bin/bash

#!/bin/bash

for i in {1001481801..1001639776}; do
    link="https://prensahistorica.mcu.es/es/catalogo_imagenes/grupo.do?path=$i"
    pdfname="$i.pdf"
    wget -q --no-check-certificate --output-document=scrapped/$pdfname $link
done
