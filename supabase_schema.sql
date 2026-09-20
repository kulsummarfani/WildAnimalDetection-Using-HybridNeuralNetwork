-- VanRakshak AI Supabase schema. Run once in the Supabase SQL editor.
-- Tables are accessed by Flask with the service key; browser roles have no table policies.

create table if not exists public.profiles (
    id uuid primary key references auth.users (id) on delete cascade,
    username text,
    role text not null default 'resident' check (role in ('admin', 'watchman', 'resident')),
    created_at timestamptz not null default now()
);

create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = ''
as $$
begin
    insert into public.profiles (id, username)
    values (new.id, split_part(coalesce(new.email, ''), '@', 1))
    on conflict (id) do nothing;
    return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
    after insert on auth.users
    for each row execute function public.handle_new_user();

insert into public.profiles (id, username)
select id, split_part(coalesce(email, ''), '@', 1) from auth.users
on conflict (id) do nothing;

create table if not exists public.alerts (
    id bigint generated always as identity primary key,
    created_at timestamptz not null default now(),
    species text not null,
    confidence real not null,
    distance_m real,
    movement text,
    threat_tier text,
    message_watchman text not null,
    message_resident text not null,
    sms_sent boolean not null default false,
    triggered_by uuid references auth.users (id) on delete set null
);
create index if not exists alerts_created_at_idx on public.alerts (created_at desc);

alter table public.profiles enable row level security;
alter table public.alerts enable row level security;
revoke all on table public.profiles from anon, authenticated;
revoke all on table public.alerts from anon, authenticated;

-- Promote users after creating them in Authentication > Users:
-- update public.profiles set role = 'admin' where id = (select id from auth.users where email = 'you@example.com');
-- update public.profiles set role = 'watchman' where id = (select id from auth.users where email = 'watchman@example.com');
