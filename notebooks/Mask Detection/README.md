# Mask Detection | Detecção de máscara

Esse notebook contém um exemplo de aplicação para detecção de máscara produzido para a disciplina de Visão Computacional 2 do Programa de Residência do CIn-UFPE.

O problema proposto foi: 

> Treinar uma rede com o dataset fornecido e usar a rede para fazer inferência em vídeo, com os *bounding boxes* e labels plotados.

Infelizmente não há como compartilhar o dataset usado (contém cerca de 400mb), mas existe um exemplo na pasta `dataset`. As informações sobre as detecções em cada imagem estão no formato PASCAL VOC XML. Foi usada a rede YOLOv5 para fazer as detecções, então esse formato precisou ser [convertido](https://towardsdatascience.com/convert-pascal-voc-xml-to-yolo-for-object-detection-f969811ccba5). Para mais informações, veja os comentários no *notebook*.


<div align="center">
  <p>⠀</p>
  <hr>
  <p>⠀</p>
  <img src="sample.gif" />
</div>