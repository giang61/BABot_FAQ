/* Délimiteur changé en ; */
/* Connexion à C:\Users\bob\PycharmProjects\QA Tool\sqlitedb\temp.db pour SQLite, nom d'utilisateur , mot de passe : Yes… */
PRAGMA busy_timeout=30000;
SELECT DATETIME();
SELECT sqlite_version();
/* Connecté. ID du processus : 14312 */
/* Reading function definitions from C:\Program Files\HeidiSQL\functions-sqlite.ini */
SELECT * FROM pragma_database_list;
/* Ouverture de la session « sqlite temp db » */
/* C:\Users\bob\AppData\Roaming\HeidiSQL\Backups\query-tab-2024-06-17_16-45-33-784.sql loaded in process #8632 */
/* Contrôles d'échelle des DPI d'écran : % */
SELECT * FROM "temp".sqlite_master WHERE type IN('table', 'view') AND name NOT LIKE 'sqlite_%';
SELECT * FROM "temp".pragma_table_info('Aprovisionnement');
/* Type de donnée inconnu «  ». Repli vers UNKNOWN. */
SELECT * FROM "temp".pragma_table_info('Aprovisionnement') WHERE pk!=0 ORDER BY pk;
SELECT * FROM "temp".pragma_index_list('Aprovisionnement') WHERE origin!='pk';
SELECT * FROM "temp".pragma_foreign_key_list('Aprovisionnement');
SELECT "sql" FROM "temp".sqlite_master WHERE "type"='table' AND name='Aprovisionnement';
/* Modification des tables restreinte. Pour les détails, consulter https://www.sqlite.org/lang_altertable.html#making_other_kinds_of_table_schema_changes */
SELECT strftime('%Y-%m', daterec) AS "Month", ltie, SUM(qterec1) AS "Weight"
FROM Aprovisionnement
WHERE strftime('%m', daterec) BETWEEN '01' AND '06'
GROUP BY Month, ltie
ORDER BY Month ASC LIMIT 5
;
/* Lignes affectées: 0  Lignes trouvées: 5  Avertissements: 0  Durée pour 1 requête: 0,000 s. (+ 0,031 s. réseau) */
SELECT strftime('%Y-%m', daterec) AS "Month", ltie, SUM(qterec1) AS "Weight"
FROM Aprovisionnement
WHERE strftime('%m', daterec) BETWEEN '01' AND '06'
GROUP BY Month, ltie
ORDER BY Month ASC;
/* Lignes affectées: 0  Lignes trouvées: 486  Avertissements: 0  Durée pour 1 requête: 0,000 s. (+ 0,031 s. réseau) */
SELECT ltie,
       SUM(CASE WHEN strftime('%m', daterec) = '01' THEN qterec1 ELSE 0 END) AS "January",
       SUM(CASE WHEN strftime('%m', daterec) = '02' THEN qterec1 ELSE 0 END) AS "February",
       SUM(CASE WHEN strftime('%m', daterec) = '03' THEN qterec1 ELSE 0 END) AS "March",
       SUM(CASE WHEN strftime('%m', daterec) = '04' THEN qterec1 ELSE 0 END) AS "April",
       SUM(CASE WHEN strftime('%m', daterec) = '05' THEN qterec1 ELSE 0 END) AS "May",
       SUM(CASE WHEN strftime('%m', daterec) = '06' THEN qterec1 ELSE 0 END) AS "June"
FROM Aprovisionnement
WHERE strftime('%m', daterec) BETWEEN '01' AND '06'
GROUP BY ltie
ORDER BY ltie ASC;
/* Lignes affectées: 0  Lignes trouvées: 169  Avertissements: 0  Durée pour 1 requête: 0,000 s. (+ 0,062 s. réseau) */