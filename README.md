# uavwaste



## Beschreibung des Projekts
Bei dem UAVVaste-Projekt geht es darum, auf Drohnenbildern zu erkennen, ob Müll vorhanden ist oder nicht. Die GPS-Daten der Bilder, auf denen Müll erkannt wurde, können an einen Aufräumroboter geschickt werden, um gegen die Umweltverschmutzung vorzugehen.
Dafür wurden die Netze VGG, DenseNet und AlexNet mit einer Hyperparametersuche mit Hilfe von Optuna trainiert und das beste Ergebnis gespeichert. 


## Datensatz
Der verwendete Datensatz ist der UAVVaste-Datensatz, herunterladbar von Github: https://github.com/UAVVaste/UAVVaste. Dabei braucht man nur den annotations-Ordner, den man in einen config-Ordner legt. Dabei kümmert sich setup.py um das Herunterladen der Bilder und deren Formatierung in die richtige Größe. Die Bilder habe ich auf 224x224-pixel große Bilder zugeschnitten, wobei diejenigen ohne Müll der Klasse Background angehören, die mit Müll der Klasse Rubbish. Die Struktur des erstellten Ordners sieht dabei folgendermaßen aus:
data
-224
--train 
---background
---rubbish
--val
---background
---rubbish
--test
---background
---rubbish
Bei dem Datensatz für das Training habe ich alle Bilder so belassen und zusätzlich augmentierte Bilder benutzt. Folgende Augmentationen wurden durchgeführt: Ein horizontales bzw. vertikales Drehen um 90 Grad und eine Invertierung je mit der Wahrscheinlichkeit 0.5, einen Gaussian-Blur mit der Kernelgröße 5 und einen Color-Jitter, alles Transformierungen aus der Bibliothek torchvision.transforms. Außerdem wurden alle Bilder in einen Tensor umgewandelt und normalisiert, bevor Sie dem Netz gefüttert wurden.

## Beschreibung des Netzes

Da bei der Hyperparametersuche AlexNet die besten Ergebnisse erzielte, beschreibe ich seinen Aufbau im Folgenden. Die erste Schicht ist eine Convolution-Schicht, gefolgt von einer Relu-Aktivierungsfunktion und anschließend einer Maxpool-schicht. Dies wird nochmal wiederholt. Danach werden dreimal wiederholt eine Convolution-Schicht gefolgt von einer Relu-Aktivierungsfunktion. Darauf folgen eine Maxpool-Schicht und eine adaptive Averagepool-Schicht. Danach folgen wieder einmal wiederholt der Dropout für die Vermeidung des Overfittings, eine Linear-Schicht und die Relu-Aktivierungsfunktion. Zum Schluss fogt eine Linear-Schicht und anschließend eine Sigmoid-Aktivierungsfunktion, da man nur zwei Klassen hat und daher die BinaryCrossEntropy als Loss anwendet. 
Alexnet verwendet 5 Convolution-Schichten und 3 voll verbundene Schichten. Nach jeder Schicht wird die ReLu-Aktivierungsfunktion angewandt, außer bei der letzen. Dort findet die Sigmoid-Aktivierungsfunktion anwendung, um den BinaryCrossEntropy-Loss verwenden zu können. Für die Vermeidung des Overfitting wird der Dropout in den ersten zwei voll verbundenen Schichten benutzt. Dabei wird mit einer gewissen Wahrscheinlichkeit die Ausgabe von jedem versteckten Neuron auf 0 gesetzt, sodass sie zu dem Forward-Pass nicht beitragen und bei der Backpropagation auch nicht teilnehmen. Diese Wahrscheinlichkeit wurde von Optuna auf 0.7 festgelegt. Nach der ersten, zweiten und fünften Convolution-Schicht wird Maxpooling angewendet, adaptive average Pooling nach dem dritten Maxpooling. Die Filtergröße in den ersten zwei Convolutional-Schichten ist 11 und 5 mit einer Schrittgröße von 4  bzw. 1 und einem Padding von je 2, in allen anderen 3 mit einem Padding von 1. Die Kernelgröße der Maxpooling-Schichten ist ebenfalls 3 mit einer Schrittgröße von 2. Die Average-Pooling-Schicht hat eine Kernelgröße von 6. Optuna hat unter der Auswahl von SGD, Adam und RMSProp SGD gewählt, wobei für die Lernrate 0.0357, für den Verfall 2.119e-07 und für das Momentum 0.3 gewählt wurde. Außerdem betrug die Batch-Size 32 (8, 16 und 24 standen auch zur Auswahl) und die Zahl der Epochen 250 (50, 100, 150, 200 und 300 standen auch zur Auswahl). Allerdings wurde durch das eingebaute EarlyStopping mit einer Patience von 17 Epochen schon nach der 35. Epoche abgebrochen. Übrigens standen für die Learning-Rate Werte zwischen 1e-5 und 1e-1 zur Auswahl, für den Verfall Werte zwischen 1e-8 und 1e-4, für das Momentum Werte zwischen 0.1 und 0.9 und für das Dropout Werte zwischen 0.2 und 0.8.


## Ergebnisse
Das erzielte Ergebnis für den Validierungsdatensatz beträgt 95.49% Accuracy, 51.28% Precision und 66.45% Recall. Dabei ist der F1-score 57.88%, der F2-score 62.73% und der F3-score 64.54%. Optimiert wurde dabei auf den F1-score, da dieser das harmonische Mittel zwischen Precision und Recall darstellt. Bei dem Testdatensatz lag die Accuracy nur wenig unter der des Validierungsdatensatzen, nämlich bei 94,27%. Der Recall lag mit 68,52% höher, aber die Precision mit 39,22% unter der des Validierungsdatensatzes. Der F1-score betrug 49,88%, der F2-score 59,61% und der F3-Score 63,75%.
