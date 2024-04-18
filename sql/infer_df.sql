WITH cte AS (
    SELECT 
        items.*,
        {{ user_id }} AS user_id
    FROM items
    WHERE items.item_id IN {{ item_ids }}
)

SELECT 
    *,
    occupation_id AS occupation,
    age_id AS age,
    COALESCE(
		(
            SELECT 
                MAX(rating_timestamp)
            FROM ratings 
            JOIN rating_phase USING (rating_id) 
            WHERE user_id = {{ user_id }} AND phase = 'train'
        ),
		(
            SELECT 
                MAX(rating_timestamp) 
            FROM ratings
        )
	) AS timestamp
FROM cte
JOIN users USING (user_id)
