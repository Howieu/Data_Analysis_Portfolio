-- 知选 app storage. Accessed only server-side with the service role; RLS on with no policies blocks anon/authenticated clients.
create table if not exists zx_users (
  id bigint generated always as identity primary key,
  email text not null unique,
  name text not null,
  pass text not null,
  created_at bigint not null
);
create table if not exists zx_sessions (
  token text primary key,
  user_id bigint references zx_users(id) on delete cascade,
  guest text not null,
  created_at bigint not null,
  expires_at bigint not null
);
create index if not exists zx_sessions_user on zx_sessions(user_id);
create table if not exists zx_favorites (
  owner text not null,
  product_id text not null,
  created_at bigint not null,
  primary key (owner, product_id)
);
create table if not exists zx_conversations (
  id text primary key,
  owner text not null,
  title text not null,
  session text not null,
  created_at bigint not null,
  updated_at bigint not null
);
create index if not exists zx_conversations_owner on zx_conversations(owner, updated_at desc);
create table if not exists zx_messages (
  id bigint generated always as identity primary key,
  conversation_id text not null references zx_conversations(id) on delete cascade,
  role text not null,
  text text not null,
  reply text,
  created_at bigint not null
);
create index if not exists zx_messages_conv on zx_messages(conversation_id, id);
create table if not exists zx_prefs (
  owner text primary key,
  active_conversation text
);
alter table zx_users enable row level security;
alter table zx_sessions enable row level security;
alter table zx_favorites enable row level security;
alter table zx_conversations enable row level security;
alter table zx_messages enable row level security;
alter table zx_prefs enable row level security;
