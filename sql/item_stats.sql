WITH item_rating_cte AS (
	SELECT 
		rating_id,
		rating
	FROM ratings
	WHERE item_id = {{ item_id }}
)
SELECT
	COUNT(*) AS num_ratings,
	ROUND(AVG(rating), 2) AS avg_ratings
FROM item_rating_cte
JOIN rating_phase USING (rating_id)
WHERE phase = 'train'
