"""Networking: a network provider's sites, devices, interfaces, traffic counters, alarms, tickets and billing.

A benchmark domain (not in the gated set): it plants what a network operator's warehouse
holds and what today's learning is known to miss, so the miss is measured, not hidden.

What this warehouse is built to test:

* devices hang off other devices (an access switch's uplink is a distribution switch, whose
  uplink is a core router): a link from the device table to itself, with core routers at
  the top having none;
* traffic and error counts are running counters sampled every 15 minutes: each sample is the
  total since the interface last restarted, and restarts set it back to zero. A counter is
  never added up across samples; utilization is a percentage;
* timestamps are in UTC; sites run in their own time zones;
* a services table dated by its contract's start and end (validity), with running contracts
  having no end;
* tickets name the circuit they are about by a text code, and 3% of them name circuits that
  were never loaded (a partly supported link);
* monthly billing is one row per service and month, an amount for the month (a flow, added
  up over months), shaped like a snapshot;
* a backup copy of the device table (devices_bak) taken three months ago sits beside it;
* devices carry a last-modified stamp (audit); speeds and bandwidths are attributes, not
  metrics.

Counters cover the last 7 days before 12 June 2026; everything else runs from January 2025.
Questions are asked on 15 June 2026.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from evals.core2.framework import DateTruth, Domain, JoinTruth, MeasureTruth, TableDef, Truth

TODAY = dt.date(2026, 6, 15)
FIRST_DAY = dt.date(2025, 1, 1)
LAST_DAY = dt.date(2026, 6, 12)
CITIES = [("Toronto", "Ontario", "America/Toronto"), ("Ottawa", "Ontario", "America/Toronto"),
          ("Montreal", "Quebec", "America/Toronto"), ("Calgary", "Alberta", "America/Edmonton"),
          ("Vancouver", "British Columbia", "America/Vancouver"), ("Halifax", "Nova Scotia", "America/Halifax"),
          ("Winnipeg", "Manitoba", "America/Winnipeg"), ("Regina", "Saskatchewan", "America/Regina")]
MODELS = {"core": ["CR-9000", "CR-9500"], "distribution": ["DS-4800", "DS-5200"], "access": ["AS-1200", "AS-1400"],
          "firewall": ["FW-700"]}
_WORDS = ["Apex", "Birch", "Cedar", "Delta", "Ember", "Fable", "Garnet", "Harbor", "Iris", "Juniper", "Kestrel",
          "Lumen", "Maple", "Nimbus", "Orchid", "Pioneer", "Quarry", "Raven", "Summit", "Tundra"]
_KINDS = ["Logistics", "Clinics", "Foods", "Legal", "Design", "Energy", "Media", "Retail", "Schools", "Labs"]


def build(seed: int = 7) -> Domain:
    rng = np.random.default_rng(seed)

    # ── sites and devices, each device hanging off its uplink ───────────────
    sites = pd.DataFrame({
        "site_id": np.arange(1, 25),
        "site_code": [f"{CITIES[i % 8][0][:3].upper()}{i // 8 + 1:02d}" for i in range(24)],
        "site_name": [f"{CITIES[i % 8][0]} {['Central', 'East', 'West'][i // 8]}" for i in range(24)],
        "region": [CITIES[i % 8][1] for i in range(24)], "timezone": [CITIES[i % 8][2] for i in range(24)]})
    devices = []
    for site in sites.itertuples():
        core = None
        if site.site_id <= 8:
            core = len(devices) + 1
            devices.append((core, f"{site.site_code.lower()}-cr01", "core", None, int(site.site_id)))
        dist = len(devices) + 1
        uplink = core or ((site.site_id - 1) % 8) * 7 + 1      # a site without a core hangs off its city's core
        devices.append((dist, f"{site.site_code.lower()}-ds01", "distribution", uplink, int(site.site_id)))
        for k in range(int(rng.integers(3, 6))):
            devices.append((len(devices) + 1, f"{site.site_code.lower()}-as{k + 1:02d}", "access", dist, int(site.site_id)))
        devices.append((len(devices) + 1, f"{site.site_code.lower()}-fw01", "firewall", dist, int(site.site_id)))
    # Cores were numbered as they were made: point every distribution uplink at a real core.
    core_of_city = {d[4]: d[0] for d in devices if d[2] == "core"}
    devices = [(i, h, r, (core_of_city[((s - 1) % 8) + 1] if r == "distribution" else u), s)
               for i, h, r, u, s in devices]
    n_dev = len(devices)
    installed = [FIRST_DAY - dt.timedelta(days=int(d)) for d in rng.integers(30, 2000, n_dev)]
    devices_frame = pd.DataFrame({
        "device_id": [d[0] for d in devices], "hostname": [d[1] for d in devices],
        "device_role": [d[2] for d in devices],
        "vendor_model": [str(rng.choice(MODELS[d[2]])) for d in devices],
        "site_id": [d[4] for d in devices],
        "uplink_device_id": pd.array([d[3] for d in devices], dtype="Int64"),
        "installed_date": installed,
        # Written by the inventory sync each night: many rows share one stamp.
        "last_modified_ts": [pd.Timestamp(LAST_DAY) - pd.Timedelta(days=int(d)) + pd.Timedelta(hours=1)
                             for d in rng.choice([0, 1, 7, 30], n_dev)]})
    backup = devices_frame.iloc[: int(n_dev * 0.93)].copy()
    backup["last_modified_ts"] = pd.Timestamp(LAST_DAY) - pd.Timedelta(days=92) + pd.Timedelta(hours=1)

    interfaces = []
    for d in devices:
        count = {"core": 8, "distribution": 6, "access": 4, "firewall": 3}[d[2]]
        speed = {"core": 100000, "distribution": 10000, "access": 1000, "firewall": 10000}[d[2]]
        for k in range(count):
            interfaces.append((len(interfaces) + 1, d[0], f"Eth1/{k + 1}", speed, "UP" if rng.random() > 0.05 else "DOWN"))
    interfaces_frame = pd.DataFrame(interfaces, columns=["interface_id", "device_id", "interface_name", "speed_mbps",
                                                         "admin_status"])

    # ── running counters every 15 minutes, for the busiest 40 interfaces ────
    watched = interfaces_frame[interfaces_frame["admin_status"] == "UP"].head(40)
    stamps = pd.date_range(pd.Timestamp(LAST_DAY) - pd.Timedelta(days=7), pd.Timestamp(LAST_DAY), freq="15min",
                           inclusive="left")
    counters = []
    for itf in watched.itertuples():
        rate = float(itf.speed_mbps) * 1e6 / 8 * 900 * float(rng.uniform(0.05, 0.45))   # bytes per 15 minutes
        in_total, out_total, errors = float(rng.uniform(1e9, 1e12)), float(rng.uniform(1e9, 1e12)), 0.0
        restart = int(rng.integers(100, len(stamps)))
        for k, ts in enumerate(stamps):
            if k == restart:
                in_total = out_total = errors = 0.0                 # the interface restarted: counters reset
            share = float(rng.uniform(0.6, 1.4))
            in_total += rate * share
            out_total += rate * share * float(rng.uniform(0.3, 0.9))
            errors += float(rng.poisson(0.2))
            counters.append((ts, int(itf.interface_id), int(in_total), int(out_total), int(errors),
                             round(min(99.9, 100.0 * rate * share / (float(itf.speed_mbps) * 1e6 / 8 * 900)), 2)))
    counters_frame = pd.DataFrame(counters, columns=["sample_ts", "interface_id", "in_octets", "out_octets",
                                                     "in_errors", "utilization_pct"])

    # ── alarms ──────────────────────────────────────────────────────────────
    total_days = (LAST_DAY - FIRST_DAY).days
    n_alarms = 4000
    raised = sorted(pd.Timestamp(FIRST_DAY) + pd.Timedelta(minutes=int(m))
                    for m in rng.integers(0, total_days * 1440, n_alarms))
    acked = [r + pd.Timedelta(minutes=int(m)) for r, m in zip(raised, rng.integers(1, 120, n_alarms))]
    cleared = [a + pd.Timedelta(minutes=int(m)) if rng.random() > 0.03 else pd.NaT
               for a, m in zip(acked, rng.integers(5, 2000, n_alarms))]
    alarms = pd.DataFrame({
        "alarm_id": np.arange(1, n_alarms + 1),
        "device_id": rng.choice(devices_frame["device_id"], n_alarms),
        "severity": rng.choice(["CRIT", "MAJ", "MIN", "WRN"], n_alarms, p=[0.08, 0.22, 0.4, 0.3]),
        "alarm_type": rng.choice(["Link down", "High CPU", "Power supply", "BGP neighbour down", "High temperature"],
                                 n_alarms),
        "raised_ts": raised, "acknowledged_ts": acked, "cleared_ts": cleared})

    # ── customers, services (contracts) and tickets ─────────────────────────
    n_cust = 180
    customers = pd.DataFrame({
        "customer_id": np.arange(1, n_cust + 1),
        "customer_name": [f"{_WORDS[i % 20]} {_KINDS[(i // 20 + i) % 10]} {i // 40 + 1}" for i in range(n_cust)],
        "segment": rng.choice(["Small business", "Enterprise", "Public sector"], n_cust, p=[0.6, 0.3, 0.1])})
    n_srv = 420
    starts = [FIRST_DAY - dt.timedelta(days=400) + dt.timedelta(days=int(d)) for d in rng.integers(0, 900, n_srv)]
    ends = []
    for s in starts:
        end = s + dt.timedelta(days=int(rng.choice([365, 730, 1095])))
        ends.append(end if end <= LAST_DAY and rng.random() < 0.7 else None)
    services = pd.DataFrame({
        "service_id": np.arange(1, n_srv + 1),
        "circuit_code": [f"CKT-{40000 + 13 * i}" for i in range(n_srv)],
        "customer_id": rng.integers(1, n_cust + 1, n_srv), "site_id": rng.integers(1, 25, n_srv),
        "service_type": rng.choice(["Internet", "MPLS", "SD-WAN", "Wavelength"], n_srv, p=[0.45, 0.25, 0.2, 0.1]),
        "bandwidth_mbps": rng.choice([100, 500, 1000, 10000], n_srv, p=[0.3, 0.3, 0.3, 0.1]),
        "contract_start": starts, "contract_end": ends})
    n_tickets = 5000
    opened = sorted(pd.Timestamp(FIRST_DAY) + pd.Timedelta(minutes=int(m))
                    for m in rng.integers(0, total_days * 1440, n_tickets))
    circuits = list(services["circuit_code"])
    named = [str(rng.choice(circuits)) if rng.random() > 0.03 else f"CKT-{90000 + int(rng.integers(0, 999))}"
             for _ in range(n_tickets)]
    ack = [o + pd.Timedelta(minutes=int(m)) for o, m in zip(opened, rng.integers(5, 240, n_tickets))]
    resolved = [a + pd.Timedelta(minutes=int(m)) for a, m in zip(ack, rng.integers(30, 4320, n_tickets))]
    resolved = [r if r <= pd.Timestamp(LAST_DAY) else pd.NaT for r in resolved]
    tickets = pd.DataFrame({
        "ticket_id": np.arange(1, n_tickets + 1), "ticket_number": [f"INC{800000 + i}" for i in range(n_tickets)],
        "circuit_code": named,
        "device_id": pd.array([int(rng.choice(devices_frame["device_id"])) if rng.random() < 0.6 else None
                               for _ in range(n_tickets)], dtype="Int64"),
        "priority": rng.choice(["P1", "P2", "P3", "P4"], n_tickets, p=[0.05, 0.2, 0.45, 0.3]),
        "category": rng.choice(["Outage", "Degraded", "Configuration", "Request"], n_tickets, p=[0.2, 0.3, 0.2, 0.3]),
        "opened_ts": opened, "acknowledged_ts": ack, "resolved_ts": resolved})

    # ── monthly billing: an amount for each month, one row per service ──────
    billing = []
    fee = {int(r.service_id): round(float(r.bandwidth_mbps) ** 0.5 * float(rng.uniform(9, 14)), 2)
           for r in services.itertuples()}
    for month in pd.date_range(FIRST_DAY, LAST_DAY, freq="MS"):
        m = month.date()
        for s in services.itertuples():
            if s.contract_start <= m and (s.contract_end is None or s.contract_end >= m):
                usage = round(float(s.bandwidth_mbps) * float(rng.uniform(0.5, 4.0)), 1)
                billing.append((m, int(s.service_id), round(fee[int(s.service_id)] * float(rng.uniform(0.98, 1.06)), 2),
                                usage))
    billing_frame = pd.DataFrame(billing, columns=["billing_month", "service_id", "billed_amount", "data_usage_gb"])

    tables = [
        TableDef("sites", "dimension", sites, {"site_id": "INTEGER", "site_code": "VARCHAR", "site_name": "VARCHAR",
                                               "region": "VARCHAR", "timezone": "VARCHAR"}, ["site_id"], "Site"),
        TableDef("devices", "dimension", devices_frame,
                 {"device_id": "INTEGER", "hostname": "VARCHAR", "device_role": "VARCHAR",
                  "vendor_model": "VARCHAR", "site_id": "INTEGER", "uplink_device_id": "INTEGER",
                  "installed_date": "DATE", "last_modified_ts": "TIMESTAMP"}, ["device_id"], "Device"),
        TableDef("devices_bak", "dimension", backup,
                 {"device_id": "INTEGER", "hostname": "VARCHAR", "device_role": "VARCHAR",
                  "vendor_model": "VARCHAR", "site_id": "INTEGER", "uplink_device_id": "INTEGER",
                  "installed_date": "DATE", "last_modified_ts": "TIMESTAMP"}, ["device_id"], "Device backup"),
        TableDef("interfaces", "dimension", interfaces_frame,
                 {"interface_id": "INTEGER", "device_id": "INTEGER", "interface_name": "VARCHAR",
                  "speed_mbps": "INTEGER", "admin_status": "VARCHAR"}, ["interface_id"], "Interface"),
        TableDef("interface_counters", "fact", counters_frame,
                 {"sample_ts": "TIMESTAMP", "interface_id": "INTEGER", "in_octets": "BIGINT",
                  "out_octets": "BIGINT", "in_errors": "BIGINT", "utilization_pct": "DECIMAL(5,2)"},
                 ["sample_ts", "interface_id"], "Interface counter"),
        TableDef("alarms", "fact", alarms,
                 {"alarm_id": "INTEGER", "device_id": "INTEGER", "severity": "VARCHAR", "alarm_type": "VARCHAR",
                  "raised_ts": "TIMESTAMP", "acknowledged_ts": "TIMESTAMP", "cleared_ts": "TIMESTAMP"},
                 ["alarm_id"], "Alarm"),
        TableDef("customers", "dimension", customers, {"customer_id": "INTEGER", "customer_name": "VARCHAR",
                                                       "segment": "VARCHAR"}, ["customer_id"], "Customer"),
        TableDef("services", "dimension", services,
                 {"service_id": "INTEGER", "circuit_code": "VARCHAR", "customer_id": "INTEGER", "site_id": "INTEGER",
                  "service_type": "VARCHAR", "bandwidth_mbps": "INTEGER", "contract_start": "DATE",
                  "contract_end": "DATE"}, ["service_id"], "Service"),
        TableDef("tickets", "fact", tickets,
                 {"ticket_id": "INTEGER", "ticket_number": "VARCHAR", "circuit_code": "VARCHAR",
                  "device_id": "INTEGER", "priority": "VARCHAR", "category": "VARCHAR", "opened_ts": "TIMESTAMP",
                  "acknowledged_ts": "TIMESTAMP", "resolved_ts": "TIMESTAMP"}, ["ticket_id"], "Ticket"),
        TableDef("service_billing_monthly", "fact", billing_frame,
                 {"billing_month": "DATE", "service_id": "INTEGER", "billed_amount": "DECIMAL(12,2)",
                  "data_usage_gb": "DECIMAL(12,1)"}, ["billing_month", "service_id"], "Service billing"),
    ]

    truth = Truth(
        kinds={t.name: t.kind for t in tables},
        primary_keys={t.name: t.primary_key for t in tables},
        joins=[
            JoinTruth("devices", "site_id", "sites", "site_id"),
            JoinTruth("devices", "uplink_device_id", "devices", "device_id", role="Uplink device"),
            JoinTruth("interfaces", "device_id", "devices", "device_id"),
            JoinTruth("interface_counters", "interface_id", "interfaces", "interface_id"),
            JoinTruth("alarms", "device_id", "devices", "device_id"),
            JoinTruth("services", "customer_id", "customers", "customer_id"),
            JoinTruth("services", "site_id", "sites", "site_id"),
            JoinTruth("tickets", "circuit_code", "services", "circuit_code", declared=False, trust="proposed"),
            JoinTruth("tickets", "device_id", "devices", "device_id"),
            JoinTruth("service_billing_monthly", "service_id", "services", "service_id"),
        ],
        calendar=None,
        dates=[
            DateTruth("devices", "installed_date", "Installed date", "event", False, default=True),
            DateTruth("devices", "last_modified_ts", "Last modified", "audit", False),
            DateTruth("interface_counters", "sample_ts", "Sample time", "event", False, default=True),
            DateTruth("alarms", "raised_ts", "Raised time", "event", False, default=True),
            DateTruth("alarms", "acknowledged_ts", "Acknowledged time", "event", False),
            DateTruth("alarms", "cleared_ts", "Cleared time", "event", False),
            DateTruth("services", "contract_start", "Contract start", "validity", False, default=True),
            DateTruth("services", "contract_end", "Contract end", "validity", False),
            DateTruth("tickets", "opened_ts", "Opened time", "event", False, default=True),
            DateTruth("tickets", "acknowledged_ts", "Acknowledged time", "event", False),
            DateTruth("tickets", "resolved_ts", "Resolved time", "event", False),
            DateTruth("service_billing_monthly", "billing_month", "Billing month", "event", False, default=True),
        ],
        measures=[
            # Running counters: the latest sample (or a difference of two) means something, a sum never does.
            MeasureTruth("interface_counters", "in_octets", "max", "non_additive", "number", "Octets in"),
            MeasureTruth("interface_counters", "out_octets", "max", "non_additive", "number", "Octets out"),
            MeasureTruth("interface_counters", "in_errors", "max", "non_additive", "number", "Input errors"),
            MeasureTruth("interface_counters", "utilization_pct", "avg", "non_additive", "percent", "Utilization"),
            MeasureTruth("alarms", None, "count", "additive", "count", "Alarms"),
            MeasureTruth("tickets", None, "count", "additive", "count", "Tickets"),
            MeasureTruth("service_billing_monthly", "billed_amount", "sum", "additive", "currency", "Billed amount"),
            MeasureTruth("service_billing_monthly", "data_usage_gb", "sum", "additive", "number", "Data usage"),
        ],
        labels={"sites": "site_name", "devices": "hostname", "customers": "customer_name"},
        codes={"sites": "site_code", "services": "circuit_code", "tickets": "ticket_number"},
        statuses={"alarms.severity": {"values": {"CRIT": "Critical", "MAJ": "Major", "MIN": "Minor", "WRN": "Warning"},
                                      "cancelled": []}},
        quality=[
            {"object": "devices_bak", "kind": "backup_copy"},
            {"object": "interface_counters.in_octets", "kind": "running_counter"},
            {"object": "devices.last_modified_ts", "kind": "load_timestamp"},
        ],
        not_measures=["interfaces.speed_mbps", "services.bandwidth_mbps", "interface_counters.interface_id",
                      "alarms.device_id", "tickets.device_id", "service_billing_monthly.service_id",
                      "devices.uplink_device_id", "devices.site_id"],
    )
    return Domain("networking", tables, truth, TODAY,
                  "A network provider's sites, devices, interface traffic, alarms, customer services, tickets and "
                  "monthly billing.",
                  abbreviations={"device": "DVC", "devices": "DVC", "interface": "INTF", "interfaces": "INTF",
                                 "counters": "CNTR", "octets": "OCT", "uplink": "UPLNK", "alarm": "ALRM",
                                 "alarms": "ALRM", "circuit": "CKT", "ticket": "TKT", "tickets": "TKT",
                                 "utilization": "UTIL", "acknowledged": "ACK", "resolved": "RSLV",
                                 "billing": "BIL", "usage": "USG", "bandwidth": "BW", "hostname": "HOST"})
