# O Projekcie

Ten projekt puszcza odtworzoną przeze mnie prostą wersję Tetrisa i oddaje
sterowanie modelowi AI, który wytrenowałem. Ten model ma nieco ułatwioną robote,
bo nie kazałem mu wykonywać każdego pojedyńczego przejścia po kratkach, tylko
wybiera sobie kolumnę na którą chciałby puścić dany klocek i to jak on ma być
obrócony. Wyobraża on sobie każdą możliwość dwóch ruchów w przyszłość i
stwierdza jak się ruszyć. Sam model nie analizuje każdej kratki na ekranie,
tylko dostaje przygotowane dane zebrane z planszy: wysokość najwyższej kolumny,
ilość dziur (pustych kratek otoczonych częściami klocków), sumę różnic wysokości
kolumn stojących obok siebie i ilość punktów jaką by dostał za dany ruch.<br>
Rok powstania projektu: 2022

# Jak go używać

Jak puści się plik "display.py" to wczytane zostaną dane modelu z folderu
"training_checkpoints" i otworzy się okno w którym będzie widać jak AI gra w
Tetrisa. Plik "training.py" posłużył do wytrenowania tego modelu. Odpalenie go
wznowiłoby proces trenowania modelu.

# Prezentacja

https://github.com/user-attachments/assets/46fd492a-750e-403e-b53a-4236b735a241
