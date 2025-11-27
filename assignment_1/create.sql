-- Create tokens table
CREATE TABLE tokens (
    token_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
);

-- Create property_groups table
CREATE TABLE property_groups (
    group_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
);

-- Create locations table
CREATE TABLE locations (
    location_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    type TEXT NOT NULL CHECK(type IN ('Building', 'Special'))
);

-- Create specials table
CREATE TABLE specials (
    location_id INTEGER PRIMARY KEY,
    description TEXT NOT NULL,
    FOREIGN KEY (location_id) REFERENCES locations(location_id)
);

-- Create players table
CREATE TABLE players (
    player_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    token_id INTEGER NOT NULL UNIQUE,
    credits INTEGER DEFAULT 0,
    current_location_id INTEGER NOT NULL,
    is_suspended INTEGER DEFAULT 0 CHECK(is_suspended IN (0, 1)),
    FOREIGN KEY (token_id) REFERENCES tokens(token_id),
    FOREIGN KEY (current_location_id) REFERENCES locations(location_id)
);

-- Create buildings table
CREATE TABLE buildings (
    location_id INTEGER PRIMARY KEY,
    group_id INTEGER NOT NULL,
    tuition_fee INTEGER NOT NULL,
    cost INTEGER NOT NULL,
    owner_id INTEGER,
    FOREIGN KEY (location_id) REFERENCES locations(location_id),
    FOREIGN KEY (group_id) REFERENCES property_groups(group_id),
    FOREIGN KEY (owner_id) REFERENCES players(player_id)
);

-- Create audit_log table
CREATE TABLE audit_log (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    round_num INTEGER NOT NULL,
    player_id INTEGER NOT NULL,
    location_id INTEGER NOT NULL,
    balance_after_turn INTEGER,
    action_type TEXT,
    FOREIGN KEY (player_id) REFERENCES players(player_id),
    FOREIGN KEY (location_id) REFERENCES locations(location_id)
);