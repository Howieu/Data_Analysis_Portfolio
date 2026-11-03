import { DatabaseSync } from 'node:sqlite';
import { mkdirSync } from 'node:fs';
import { dirname } from 'node:path';

export function openDb(file) {
  if (file !== ':memory:') mkdirSync(dirname(file), { recursive: true });
  const db = new DatabaseSync(file);
  db.exec(`
    pragma journal_mode = wal;
    pragma foreign_keys = on;
    create table if not exists users (id integer primary key, email text not null unique, name text not null, pass text not null, created_at integer not null);
    create table if not exists sessions (token text primary key, user_id integer references users(id) on delete cascade, guest text not null, created_at integer not null, expires_at integer not null);
    create table if not exists favorites (owner text not null, product_id text not null, created_at integer not null, primary key (owner, product_id));
    create table if not exists conversations (id text primary key, owner text not null, title text not null, session text not null, created_at integer not null, updated_at integer not null);
    create index if not exists conv_owner on conversations(owner, updated_at desc);
    create table if not exists messages (id integer primary key, conversation_id text not null references conversations(id) on delete cascade, role text not null, text text not null, reply text, created_at integer not null);
    create table if not exists prefs (owner text primary key, active_conversation text);
  `);
  return db;
}
