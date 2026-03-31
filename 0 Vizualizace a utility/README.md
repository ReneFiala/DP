**cloud_checker.py**

Kontrola souborů anotovaných PCD souborů s možností automatické detekce duplikátů či nepoměrně malých/velkých anotací či manuálního označení vzorků k další kontrole. Použito v rámci kontroly datového souboru. Možnost jít tam a zpátky - viz výstup v konzoli pro nápovědu ovládání a -h pro argumenty. Vyžaduje Open3D.

**visualise_results.py**

Zobrazení predikcí uložených v predictions.pkl a porovnání s ground truths. Konfigurace pouze v rámci zdrojového souboru. Nelze předčasně ukončit ani jít tam a zpět. Vyžaduje Open3D.

**pointplot.py**

Sestavení bodového grafu (obr. 34 v práci).