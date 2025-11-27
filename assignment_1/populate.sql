-- 1. Insert Reference Data
INSERT INTO tokens (name) VALUES 
('Mortarboard'), ('Book'), ('Certificate'), ('Gown'), ('Laptop'), ('Pen');

INSERT INTO property_groups (name) VALUES 
('Green'), ('Orange'), ('Blue'), ('Brown'), ('Grey'), ('Black');

-- 2. Insert All Locations
INSERT INTO locations (location_id, name, type) VALUES
(0, 'Welcome Week', 'Special'),
(1, 'Kilburn', 'Building'),
(2, 'IT', 'Building'),
(3, 'Hearing 1', 'Special'),
(4, 'Uni Place', 'Building'),
(5, 'AMBS', 'Building'),
(6, 'RAG 1', 'Special'),
(7, 'Suspension', 'Special'),
(8, 'Crawford', 'Building'),
(9, 'Sugden', 'Building'),
(10, 'Ali G', 'Special'),
(11, 'Shopping Precinct', 'Building'),
(12, 'MECD', 'Building'),
(13, 'RAG 2', 'Special'),
(14, 'Library', 'Building'),
(15, 'Sam Alex', 'Building'),
(16, 'Hearing 2', 'Special'),
(17, 'Your''e Suspended', 'Special'),
(18, 'Museum', 'Building'),
(19, 'Whitworth Hall', 'Building');

-- 3. Insert Special Location Descriptions
INSERT INTO specials (location_id, description) VALUES
(0, 'Collect 100 credits as you pass.'),
(3, 'You are found guilty of academic malpractice. Fined 30 credits.'),
(6, 'You win a fancy dress contest. Win 75 credits.'),
(7, 'Just visiting / In suspension.'),
(10, 'Free resting.'),
(13, 'You receive a bursary and share it with your friends. All players receive 50 credits.'),
(16, 'You are in rent arrears. Fined 100 credits.'),
(17, 'Move to suspension immediately.');

-- 4. Insert Players
INSERT INTO players (player_id, name, token_id, credits, current_location_id, is_suspended) VALUES
(1, 'Gareth', 3, 430, 18, 0),
(2, 'Stewart', 1, 360, 1, 0),
(3, 'Emma', 2, 470, 5, 0),
(4, 'Nadine', 6, 400, 3, 0);

-- 5. Insert Building Details
INSERT INTO buildings (location_id, group_id, tuition_fee, cost, owner_id) VALUES
(1, 1, 15, 30, 4),
(2, 1, 15, 30, 1);

INSERT INTO buildings (location_id, group_id, tuition_fee, cost, owner_id) VALUES
(4, 2, 25, 50, 1),
(5, 2, 25, 50, 2);

INSERT INTO buildings (location_id, group_id, tuition_fee, cost, owner_id) VALUES
(8, 3, 30, 60, 3),
(9, 3, 30, 60, 1);

INSERT INTO buildings (location_id, group_id, tuition_fee, cost, owner_id) VALUES
(11, 4, 35, 70, NULL),
(12, 4, 35, 70, 2);

INSERT INTO buildings (location_id, group_id, tuition_fee, cost, owner_id) VALUES
(14, 5, 40, 80, 3),
(15, 5, 40, 80, NULL);

INSERT INTO buildings (location_id, group_id, tuition_fee, cost, owner_id) VALUES
(18, 6, 50, 100, 3),
(19, 6, 50, 100, 4);