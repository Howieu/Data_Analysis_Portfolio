-- Gareth Round 1

-- 1. Move & Pass
UPDATE players
SET 
    credits = credits + CASE 
        WHEN (current_location_id + 3) >= 20 THEN 100 
        ELSE 0 
    END,
    current_location_id = (current_location_id + 3) % 20
WHERE name = 'Gareth';

-- 2. Jail Check
UPDATE players 
SET current_location_id = 7, is_suspended = 1 
WHERE name = 'Gareth' AND current_location_id = 17;

-- 3. Try Buy
UPDATE players 
SET credits = credits - (
    SELECT cost FROM buildings 
    WHERE location_id = (SELECT current_location_id FROM players WHERE name = 'Gareth')
)
WHERE name = 'Gareth' 
  AND EXISTS (
    SELECT 1 FROM buildings 
    WHERE location_id = (SELECT current_location_id FROM players WHERE name = 'Gareth') 
      AND owner_id IS NULL
);

UPDATE buildings 
SET owner_id = (SELECT player_id FROM players WHERE name = 'Gareth') 
WHERE location_id = (SELECT current_location_id FROM players WHERE name = 'Gareth') 
  AND owner_id IS NULL;

-- 4. Try Pay Rent
UPDATE players 
SET credits = credits - COALESCE((
    SELECT 
        CASE 
            -- Check Monopoly: If owner's count == total group count -> Double Rent
            WHEN (SELECT COUNT(*) FROM buildings b2 WHERE b2.group_id = b1.group_id AND b2.owner_id = b1.owner_id) = 
                 (SELECT COUNT(*) FROM buildings b3 WHERE b3.group_id = b1.group_id) 
            THEN b1.tuition_fee * 2 
            ELSE b1.tuition_fee 
        END
    FROM buildings b1
    WHERE location_id = (SELECT current_location_id FROM players WHERE name = 'Gareth')
      AND owner_id IS NOT NULL 
      AND owner_id != (SELECT player_id FROM players WHERE name = 'Gareth')
), 0)
WHERE name = 'Gareth';

-- 5. Owner Receives Rent
UPDATE players 
SET credits = credits + COALESCE((
    SELECT 
        CASE 
            WHEN (SELECT COUNT(*) FROM buildings b2 WHERE b2.group_id = b1.group_id AND b2.owner_id = b1.owner_id) = 
                 (SELECT COUNT(*) FROM buildings b3 WHERE b3.group_id = b1.group_id) 
            THEN b1.tuition_fee * 2 
            ELSE b1.tuition_fee 
        END
    FROM buildings b1
    WHERE location_id = (SELECT current_location_id FROM players WHERE name = 'Gareth')
), 0)
WHERE player_id = (
    SELECT owner_id FROM buildings 
    WHERE location_id = (SELECT current_location_id FROM players WHERE name = 'Gareth')
      AND owner_id IS NOT NULL 
      AND owner_id != (SELECT player_id FROM players WHERE name = 'Gareth')
);

-- 6. Try Special Events
UPDATE players 
SET credits = credits + CASE current_location_id 
    WHEN 3 THEN -30 
    WHEN 16 THEN -100 
    WHEN 6 THEN 75 
    ELSE 0 
END
WHERE name = 'Gareth' AND current_location_id IN (3, 16, 6);

-- 7. Log
INSERT INTO audit_log (round_num, player_id, location_id, balance_after_turn, action_type)
SELECT 1, player_id, current_location_id, credits, 'TURN_END' 
FROM players WHERE name = 'Gareth';