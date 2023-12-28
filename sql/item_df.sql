SELECT
    items.item_id,
    movie_title,
    movie_genres,
    COALESCE(num_ratings,0) AS num_ratings,
    COALESCE(avg_ratings,0) AS avg_ratings,
    COALESCE(pos_ratings,0) AS pos_ratings,
    COALESCE(neg_ratings,0) AS neg_ratings
FROM items
LEFT JOIN (
    SELECT 
        item_id,
        COUNT(*) AS num_ratings,
        ROUND(AVG(rating), 2) AS avg_ratings,
        SUM(CASE WHEN rating > 3 THEN 1 ELSE 0 END) AS pos_ratings,
        SUM(CASE WHEN rating <= 3 THEN 1 ELSE 0 END) AS neg_ratings
    FROM ratings
    GROUP BY item_id
) item_rating USING (item_id)