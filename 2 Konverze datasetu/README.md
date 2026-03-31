**crossval2pcdet.py**  
Převádí vstupní .csv soubory splitů a složku s datasetem do formátu, přijatelný OpenPCDetem. Použijte -h pro nápovědu. Vyžaduje Open3D. Po uložení do podsložky v OpenPCDet/data je ještě potřeba spustit skript pcdet/datasets/custom/custom_dataset.py, pro nested je dostupný bash skript nested_custom_infos.sh v rootu OpenPCDet.

**csv/ **
Předvypočtené permutace CV setů se splity - 4 náhodná rozdělení, 17 vlastních metod

**dataset/ **  
Samotný datový soubor. Obsahuje 258 vzorků, 2 jsou nepoužity.