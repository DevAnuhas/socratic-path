-- SocraticPath: Initial database schema
-- Run in Supabase SQL Editor (Dashboard → SQL Editor → New Query)

-- ============================================================
-- 1. PROFILES TABLE (linked to Supabase auth.users)
-- ============================================================

create table public.profiles (
  id          uuid primary key references auth.users(id) on delete cascade,
  display_name text,
  avatar_url  text,
  email       text,
  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now()
);

alter table public.profiles enable row level security;

-- Users can only read/update their own profile
create policy "Users can read own profile"
  on public.profiles for select
  using (auth.uid() = id);

create policy "Users can update own profile"
  on public.profiles for update
  using (auth.uid() = id);

-- ============================================================
-- 2. EXPLORATIONS TABLE (session metadata)
-- ============================================================

create table public.explorations (
  id          uuid primary key default gen_random_uuid(),
  user_id     uuid not null references public.profiles(id) on delete cascade,
  title       text not null,
  root_node_id text not null,
  node_count  integer not null default 0,
  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now()
);

create index idx_explorations_user_id on public.explorations(user_id);

alter table public.explorations enable row level security;

create policy "Users can read own explorations"
  on public.explorations for select
  using (auth.uid() = user_id);

create policy "Users can insert own explorations"
  on public.explorations for insert
  with check (auth.uid() = user_id);

create policy "Users can update own explorations"
  on public.explorations for update
  using (auth.uid() = user_id);

create policy "Users can delete own explorations"
  on public.explorations for delete
  using (auth.uid() = user_id);

-- ============================================================
-- 3. EXPLORATION_NODES TABLE (tree nodes, normalised)
-- ============================================================

create table public.exploration_nodes (
  id              serial primary key,
  exploration_id  uuid not null references public.explorations(id) on delete cascade,
  node_id         text not null,
  node_type       text not null check (node_type in ('input', 'question', 'reflection')),
  text            text not null,
  parent_node_id  text,          -- null for root node
  depth           integer not null default 0,
  metadata        jsonb not null default '{}',
  children        text[] not null default '{}',
  sort_order      integer not null default 0,
  created_at      timestamptz not null default now(),

  unique (exploration_id, node_id)
);

create index idx_exploration_nodes_exploration_id on public.exploration_nodes(exploration_id);

alter table public.exploration_nodes enable row level security;

-- Nodes inherit access from their parent exploration
create policy "Users can read own exploration nodes"
  on public.exploration_nodes for select
  using (
    exists (
      select 1 from public.explorations
      where explorations.id = exploration_nodes.exploration_id
        and explorations.user_id = auth.uid()
    )
  );

create policy "Users can insert own exploration nodes"
  on public.exploration_nodes for insert
  with check (
    exists (
      select 1 from public.explorations
      where explorations.id = exploration_nodes.exploration_id
        and explorations.user_id = auth.uid()
    )
  );

create policy "Users can update own exploration nodes"
  on public.exploration_nodes for update
  using (
    exists (
      select 1 from public.explorations
      where explorations.id = exploration_nodes.exploration_id
        and explorations.user_id = auth.uid()
    )
  );

create policy "Users can delete own exploration nodes"
  on public.exploration_nodes for delete
  using (
    exists (
      select 1 from public.explorations
      where explorations.id = exploration_nodes.exploration_id
        and explorations.user_id = auth.uid()
    )
  );

-- ============================================================
-- 4. TRIGGERS
-- ============================================================

-- Auto-create profile on user signup
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer set search_path = ''
as $$
begin
  insert into public.profiles (id, display_name, avatar_url, email)
  values (
    new.id,
    coalesce(new.raw_user_meta_data ->> 'full_name', new.raw_user_meta_data ->> 'name'),
    new.raw_user_meta_data ->> 'avatar_url',
    new.email
  );
  return new;
end;
$$;

create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();

-- Auto-update updated_at timestamp
create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

create trigger set_profiles_updated_at
  before update on public.profiles
  for each row execute function public.set_updated_at();

create trigger set_explorations_updated_at
  before update on public.explorations
  for each row execute function public.set_updated_at();
