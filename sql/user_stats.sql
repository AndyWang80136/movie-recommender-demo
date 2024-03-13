SELECT 
    train_count,
    val_count,
    test_count
FROM user_phase_count
WHERE user_id = {{ user_id }}