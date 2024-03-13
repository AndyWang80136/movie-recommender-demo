SELECT 
	COUNT(*) AS num_ratings,
	ROUND(AVG(rating), 2) AS avg_ratings,
	SUM(CASE WHEN rating > 3 THEN 1 ELSE 0 END) AS pos_ratings,
	SUM(CASE WHEN rating <= 3 THEN 1 ELSE 0 END) AS neg_ratings
FROM ratings
WHERE item_id = {{ item_id }}