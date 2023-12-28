WITH cte AS (
    SELECT 
        item1_id,
        item2_id,
        similarity,
        RANK() OVER (PARTITION BY item1_id ORDER BY similarity DESC) AS sim_rank
    FROM item_genre_cossim
    WHERE item1_id = {{ item_id }}
), user_item_ratings AS (
    SELECT
        items.item_id,
        movie_title,
        movie_genres,
        COALESCE(num_ratings,0) AS num_ratings,
        COALESCE(avg_ratings,0) AS avg_ratings
    FROM items
    LEFT JOIN (
        SELECT 
            item_id,
            COUNT(*) AS num_ratings,
            ROUND(AVG(rating), 2) AS avg_ratings
        FROM ratings
        GROUP BY item_id
    ) item_rating ON item_rating.item_id = items.item_id
)
SELECT 
    item1_id,
    item2_id, 
    similarity,
    sim_rank,
    num_ratings,
    avg_ratings
FROM cte
JOIN user_item_ratings ON user_item_ratings.item_id = item2_id
ORDER BY similarity DESC, num_ratings DESC, avg_ratings DESC
LIMIT {{ num_candidates }}
