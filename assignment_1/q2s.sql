-- Stewart Round 2

-- 1. Move
UPDATE players
SET credits = credits + CASE WHEN (current_location_id + 1) >= 20 THEN 100 ELSE 0 END,
    current_location_id = (current_location_id + 1) % 20
WHERE name = 'Stewart';

-- 2. Jail Check
UPDATE players 
SET current_location_id = 7, is_suspended = 1 
WHERE name = 'Stewart' AND current_location_id = 17;

-- 3. Buy
UPDATE players 
SET credits = credits - (
    SELECT cost FROM buildings 
    WHERE location_id = (
        SELECT current_location_id FROM players WHERE name = 'Stewart'
    )
)
WHERE name = 'Stewart' AND EXISTS (
    SELECT 1 FROM buildings WHERE location_id = (
        SELECT current_location_id FROM players WHERE name = 'Stewart'
    ) AND owner_id IS NULL
);

UPDATE buildings 
SET owner_id = (SELECT player_id FROM players WHERE name = 'Stewart') 
WHERE location_id = (
    SELECT current_location_id FROM players WHERE name = 'Stewart'
) AND owner_id IS NULL;

-- 4. Rent
UPDATE players 
SET credits = credits - COALESCE((SELECT CASE WHEN (
    SELECT COUNT(*) 
    FROM buildings b2 
    WHERE b2.group_id = b1.group_id AND b2.owner_id = b1.owner_id
) = (
    SELECT COUNT(*) 
    FROM buildings b3 
    WHERE b3.group_id = b1.group_id
) THEN b1.tuition_fee*2 ELSE b1.tuition_fee END 
FROM buildings b1 
WHERE location_id = (
    SELECT current_location_id FROM players WHERE name = 'Stewart'
) AND owner_id IS NOT NULL AND owner_id != (SELECT player_id FROM players WHERE name = 'Stewart')
), 0) 
WHERE name = 'Stewart';
UPDATE players 
SET credits = credits + COALESCE((SELECT CASE WHEN (
    SELECT COUNT(*) 
    FROM buildings b2 
    WHERE b2.group_id = b1.group_id AND b2.owner_id = b1.owner_id
) = (
    SELECT COUNT(*) 
    FROM buildings b3 
    WHERE b3.group_id = b1.group_id
) THEN b1.tuition_fee*2 ELSE b1.tuition_fee END 
FROM buildings b1 
WHERE location_id = (
    SELECT current_location_id FROM players WHERE name = 'Stewart'
)), 0) 
WHERE player_id = (
    SELECT owner_id FROM buildings WHERE location_id = (
        SELECT current_location_id FROM players WHERE name = 'Stewart'
    ) AND owner_id IS NOT NULL AND owner_id != (SELECT player_id FROM players WHERE name = 'Stewart')
);

-- 5. Events
UPDATE players 
SET credits = credits + CASE 
    current_location_id WHEN 3 THEN -30 WHEN 16 THEN -100 WHEN 6 THEN 75 ELSE 0 END 
WHERE name = 'Stewart' AND current_location_id IN (3, 16, 6);

INSERT INTO audit_log (round_num, player_id, location_id, balance_after_turn, action_type)
SELECT 2, player_id, current_location_id, credits, 'TURN_END' 
FROM players WHERE name = 'Stewart';