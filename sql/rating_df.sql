WITH cte AS (
	SELECT 
		* 
	FROM rating_phase
	WHERE phase = '{{ phase }}'
)
SELECT 
	* 
FROM cte
JOIN ratings USING (rating_id)
