/*
    QueryBot EMCO-shaped Infor M3 data-mart test model  —  EMDW_DMART
    ------------------------------------------------------------------
    WARNING: drops and recreates objects in EMDW_DMART. Run only in a
    dedicated TEST database. It never touches other schemas.

    All customers, items, orders, invoices and amounts seeded by the companion
    scripts are synthetic and must not be treated as real EMCO records.

    WHY THIS SHAPE
    ==============
    This mirrors the structural characteristics of the client's real mart, which
    is what actually breaks the bot — not the volume. Each one below has caused a
    live defect:

    1. YYYYMMDD INTEGER DATE KEYS.  DT_DMS_KEY is 20260630, not an opaque
       surrogate. MAX(fact.<date>_DT_DMS_KEY) therefore equals MAX(DT_DMS.DMS_DT)
       as a number, and a date window can in principle be an integer range on the
       fact with no dimension join at all. The system currently records these as
       "surrogate_fk" and joins DT_DMS to filter dates it could filter directly.

    2. ROLE-PLAYING DATES.  CUS_ORD_IVC_FCT carries EIGHT *_DT_DMS_KEY columns,
       all pointing at DT_DMS. That is what makes "revenue for the last 2 days"
       ambiguous, and it is why an audit date (LST_MOD_DT_DMS_KEY) could be
       selected in place of the business date (CUS_IVC_DT_DMS_KEY).

    3. AUDIT DATES ON DIMENSIONS.  CUS_DMS.CR_LMT_1_CHG_DT_DMS_KEY and friends
       are date-role shaped but nobody will ever ask a question about them. They
       are the noise that inflates the entity graph.

    4. FOUR FACTS SHARING DIMENSIONS.  Invoices, purchase receipts, finance and
       item balances all reach PFT_CTR_DMS and ITM_DMS, so a loosely matched
       metric can drag two rival facts into one plan.

    5. M3 ABBREVIATED NAMING.  CUS/ORD/IVC/PCH/RCT/FNN/ITM/BAL/PRD/DT/DMS/FCT.
       Business language has to come from the semantic layer; the physical names
       carry almost none.

    6. CAD CURRENCY QUALIFIER.  Amount columns exist in both CAD and local
       currency, so picking the wrong one is silently wrong rather than an error.

    7. NO INDEXES beyond the primary keys, deliberately — the client cannot add
       them, so the test model should not have them either.
*/

SET NOCOUNT ON;
SET XACT_ABORT ON;

IF SCHEMA_ID(N'EMDW_DMART') IS NULL
    EXEC(N'CREATE SCHEMA EMDW_DMART AUTHORIZATION dbo;');

/* Drop facts before dimensions so the script is safely repeatable. */
DROP TABLE IF EXISTS EMDW_DMART.CUS_ORD_IVC_FCT;
DROP TABLE IF EXISTS EMDW_DMART.PCH_ORD_RCT_FCT;
DROP TABLE IF EXISTS EMDW_DMART.FNN_FCT;
DROP TABLE IF EXISTS EMDW_DMART.ITM_BAL_PRD_FCT;
DROP TABLE IF EXISTS EMDW_DMART.PFT_CTR_CUS_DAT;
DROP TABLE IF EXISTS EMDW_DMART.CUS_DMS;
DROP TABLE IF EXISTS EMDW_DMART.CUS_TYP_DMS;
DROP TABLE IF EXISTS EMDW_DMART.CUS_SEG_DMS;
DROP TABLE IF EXISTS EMDW_DMART.ITM_DMS;
DROP TABLE IF EXISTS EMDW_DMART.WHS_DMS;
DROP TABLE IF EXISTS EMDW_DMART.PFT_CTR_DMS;
DROP TABLE IF EXISTS EMDW_DMART.PC_DVN_DMS;
DROP TABLE IF EXISTS EMDW_DMART.SUP_DMS;
DROP TABLE IF EXISTS EMDW_DMART.DT_DMS;

/* ══════════════════════════════════════════════════════════════════════════
   DATE DIMENSION — the smart key. DT_DMS_KEY IS the date, as YYYYMMDD.
   ══════════════════════════════════════════════════════════════════════════ */
CREATE TABLE EMDW_DMART.DT_DMS (
    DT_DMS_KEY          int          NOT NULL,   /* 20260630 — YYYYMMDD, not opaque */
    DMS_DT              date         NOT NULL,   /* the calendar date value        */
    DAY                 date         NOT NULL,   /* alias column the mart also carries */
    DAY_OF_WK_NO        tinyint      NOT NULL,
    DAY_NM              nvarchar(20) NOT NULL,
    DAY_OF_MTH          tinyint      NOT NULL,
    WK_OF_YR            tinyint      NOT NULL,
    MTH_NO              tinyint      NOT NULL,
    MTH_NM              nvarchar(20) NOT NULL,
    MTH_START_DT        date         NOT NULL,
    YR_MTH_NO           int          NOT NULL,   /* 202606 — YYYYMM integer        */
    QTR_NO              tinyint      NOT NULL,
    QTR_NM              nvarchar(8)  NOT NULL,
    CAL_YR              smallint     NOT NULL,
    FSC_MTH_NO          tinyint      NOT NULL,
    FSC_QTR_NO          tinyint      NOT NULL,
    FSC_YR              smallint     NOT NULL,
    IS_WKD_FLG          bit          NOT NULL,
    CONSTRAINT PK_DT_DMS PRIMARY KEY CLUSTERED (DT_DMS_KEY)
);

/* ══════════════════════════════════════════════════════════════════════════
   DIMENSIONS
   ══════════════════════════════════════════════════════════════════════════ */
CREATE TABLE EMDW_DMART.PC_DVN_DMS (
    PC_DVN_DMS_KEY      int           NOT NULL,
    PC_DVN_CD           nvarchar(10)  NOT NULL,
    PC_DVN_NM           nvarchar(60)  NOT NULL,
    ACT_FLG             bit           NOT NULL,
    AZ_LST_UPD_TS       datetime2(0)  NULL,
    CONSTRAINT PK_PC_DVN_DMS PRIMARY KEY CLUSTERED (PC_DVN_DMS_KEY)
);

CREATE TABLE EMDW_DMART.PFT_CTR_DMS (
    PFT_CTR_DMS_KEY     int           NOT NULL,
    PFT_CTR_CD          nvarchar(10)  NOT NULL,
    PFT_CTR_NM          nvarchar(80)  NOT NULL,   /* the display dimension        */
    PC_DVN_DMS_KEY      int           NULL,
    RGN_NM              nvarchar(40)  NULL,       /* "region" questions land here */
    CNY_CD              nvarchar(4)   NULL,
    ACT_FLG             bit           NOT NULL,
    AZ_LST_UPD_TS       datetime2(0)  NULL,
    AZ_LST_UPD_USR      nvarchar(40)  NULL,
    CONSTRAINT PK_PFT_CTR_DMS PRIMARY KEY CLUSTERED (PFT_CTR_DMS_KEY)
);

CREATE TABLE EMDW_DMART.CUS_SEG_DMS (
    CUS_SEG_DMS_KEY     int           NOT NULL,
    CUS_SEG_CD          nvarchar(10)  NOT NULL,
    CUS_SEG_NM          nvarchar(60)  NOT NULL,
    CONSTRAINT PK_CUS_SEG_DMS PRIMARY KEY CLUSTERED (CUS_SEG_DMS_KEY)
);

CREATE TABLE EMDW_DMART.CUS_TYP_DMS (
    CUS_TYP_DMS_KEY     int           NOT NULL,
    CUS_TYP_CD          nvarchar(10)  NOT NULL,
    CUS_TYP_NM          nvarchar(60)  NOT NULL,
    ACT_FLG             bit           NOT NULL,
    CONSTRAINT PK_CUS_TYP_DMS PRIMARY KEY CLUSTERED (CUS_TYP_DMS_KEY)
);

/*  CUS_DMS carries the audit-date noise on purpose. CR_LMT_n_CHG_DT_DMS_KEY is
    date-role shaped and joins DT_DMS, but no business user will ever ask about
    it — it is exactly the kind of column that inflated the graph to ~20 date
    entities and competed with Invoice Date. */
CREATE TABLE EMDW_DMART.CUS_DMS (
    CUS_DMS_KEY             int           NOT NULL,
    CUS_CD                  nvarchar(12)  NOT NULL,
    CUS_NM                  nvarchar(120) NOT NULL,   /* the display dimension */
    CUS_TYP_DMS_KEY         int           NULL,
    CUS_SEG_DMS_KEY         int           NULL,
    PFT_CTR_DMS_KEY         int           NULL,       /* second path to profit centre */
    CTY_NM                  nvarchar(60)  NULL,
    PRV_CD                  nvarchar(4)   NULL,
    PST_CD                  nvarchar(12)  NULL,
    CNY_CD                  nvarchar(4)   NULL,
    CR_LMT_1                decimal(18,2) NULL,
    CR_LMT_1_CHG_DT_DMS_KEY int           NULL,       /* audit date — noise */
    CR_LMT_2                decimal(18,2) NULL,
    CR_LMT_2_CHG_DT_DMS_KEY int           NULL,       /* audit date — noise */
    CR_LMT_3                decimal(18,2) NULL,
    CR_LMT_3_CHG_DT_DMS_KEY int           NULL,       /* audit date — noise */
    FST_IVC_DT_DMS_KEY      int           NULL,       /* "customer first invoice date" */
    ACT_FLG                 bit           NOT NULL,
    AZ_LST_UPD_TS           datetime2(0)  NULL,       /* infrastructure — exclude */
    AZ_LST_UPD_USR          nvarchar(40)  NULL,       /* infrastructure — exclude */
    AZ_EXT_ID               nvarchar(40)  NULL,       /* infrastructure — exclude */
    CONSTRAINT PK_CUS_DMS PRIMARY KEY CLUSTERED (CUS_DMS_KEY)
);

CREATE TABLE EMDW_DMART.ITM_DMS (
    ITM_DMS_KEY         int           NOT NULL,
    ITM_CD              nvarchar(20)  NOT NULL,
    ITM_NM              nvarchar(120) NOT NULL,
    ITM_GRP_CD          nvarchar(10)  NULL,
    ITM_GRP_NM          nvarchar(60)  NULL,
    UOM_CD              nvarchar(6)   NULL,
    ACT_FLG             bit           NOT NULL,
    AZ_LST_UPD_TS       datetime2(0)  NULL,
    CONSTRAINT PK_ITM_DMS PRIMARY KEY CLUSTERED (ITM_DMS_KEY)
);

CREATE TABLE EMDW_DMART.WHS_DMS (
    WHS_DMS_KEY         int           NOT NULL,
    WHS_CD              nvarchar(10)  NOT NULL,
    WHS_NM              nvarchar(80)  NOT NULL,
    PFT_CTR_DMS_KEY     int           NULL,
    CTY_NM              nvarchar(60)  NULL,
    PRV_CD              nvarchar(4)   NULL,
    ACT_FLG             bit           NOT NULL,
    CONSTRAINT PK_WHS_DMS PRIMARY KEY CLUSTERED (WHS_DMS_KEY)
);

CREATE TABLE EMDW_DMART.SUP_DMS (
    SUP_DMS_KEY         int           NOT NULL,
    SUP_CD              nvarchar(12)  NOT NULL,
    SUP_NM              nvarchar(120) NOT NULL,
    CNY_CD              nvarchar(4)   NULL,
    ACT_FLG             bit           NOT NULL,
    CONSTRAINT PK_SUP_DMS PRIMARY KEY CLUSTERED (SUP_DMS_KEY)
);

/*  A second, narrow bridge from profit centre to customer. This exists so the
    pathfinder genuinely has two ways to reach a customer from an invoice —
    directly, or through the profit centre — which is the "more than one equally
    governed relationship path" case. */
CREATE TABLE EMDW_DMART.PFT_CTR_CUS_DAT (
    PFT_CTR_DMS_KEY     int           NOT NULL,
    CUS_DMS_KEY         int           NOT NULL,
    ASG_DT_DMS_KEY      int           NULL,
    PRY_FLG             bit           NOT NULL,
    ACT_FLG             bit           NOT NULL,
    AZ_LST_UPD_TS       datetime2(0)  NULL,
    CONSTRAINT PK_PFT_CTR_CUS_DAT PRIMARY KEY CLUSTERED (PFT_CTR_DMS_KEY, CUS_DMS_KEY)
);

/* ══════════════════════════════════════════════════════════════════════════
   FACT 1 — CUSTOMER ORDER INVOICE. The revenue fact.
   EIGHT role-playing dates into DT_DMS. This is the table every date-role
   defect has surfaced on.
   ══════════════════════════════════════════════════════════════════════════ */
CREATE TABLE EMDW_DMART.CUS_ORD_IVC_FCT (
    CUS_ORD_IVC_FCT_KEY     bigint        NOT NULL,
    IVC_NO                  nvarchar(20)  NOT NULL,
    IVC_LIN_NO              int           NOT NULL,
    ORD_NO                  nvarchar(20)  NULL,
    CUS_DMS_KEY             int           NULL,
    ITM_DMS_KEY             int           NULL,
    WHS_DMS_KEY             int           NULL,
    PFT_CTR_DMS_KEY         int           NULL,

    /* ── Role-playing dates. All eight join DT_DMS.DT_DMS_KEY ───────────── */
    CUS_IVC_DT_DMS_KEY      int           NULL,  /* Invoice Date   — the DEFAULT */
    CUS_ORD_DT_DMS_KEY      int           NULL,  /* Order Date                   */
    CFM_DLY_DT_DMS_KEY      int           NULL,  /* Confirmed Delivery Date      */
    RQS_DLY_DT_DMS_KEY      int           NULL,  /* Requested Delivery Date      */
    PLN_DLY_DT_DMS_KEY      int           NULL,  /* Planned Delivery Date        */
    CNL_ORD_DT_DMS_KEY      int           NULL,  /* Cancelled Order Date         */
    DUE_DT_DMS_KEY          int           NULL,  /* Due Date                     */
    LST_MOD_DT_DMS_KEY      int           NULL,  /* Last Modified Date — AUDIT   */

    /* ── Measures. Note the CAD / local currency pair ────────────────────── */
    SOP_CUS_IVC_LIN_AMT     decimal(18,2) NULL,  /* net revenue, local currency  */
    SOP_CUS_IVC_LIN_CAD_AMT decimal(18,2) NULL,  /* net revenue, Canadian dollars*/
    IVC_QTY                 decimal(18,3) NULL,
    IVC_GRS_AMT             decimal(18,2) NULL,
    IVC_DSC_AMT             decimal(18,2) NULL,
    IVC_TAX_AMT             decimal(18,2) NULL,
    IVC_CST_AMT             decimal(18,2) NULL,
    IVC_MGN_AMT             decimal(18,2) NULL,
    CNY_CD                  nvarchar(4)   NULL,
    IVC_STS_CD              nvarchar(10)  NULL,
    CNL_FLG                 bit           NOT NULL,
    AZ_LST_UPD_TS           datetime2(0)  NULL,
    CONSTRAINT PK_CUS_ORD_IVC_FCT PRIMARY KEY CLUSTERED (CUS_ORD_IVC_FCT_KEY)
);

/* ══════════════════════════════════════════════════════════════════════════
   FACT 2 — PURCHASE ORDER RECEIPT.
   The fact a "purchase orders" question must resolve to, and the one that was
   losing to CUS_ORD_IVC_FCT because a loosely matched metric named the latter.
   ══════════════════════════════════════════════════════════════════════════ */
CREATE TABLE EMDW_DMART.PCH_ORD_RCT_FCT (
    PCH_ORD_RCT_FCT_KEY     bigint        NOT NULL,
    PCH_ORD_NO              nvarchar(20)  NOT NULL,
    PCH_ORD_LIN_NO          int           NOT NULL,
    SUP_DMS_KEY             int           NULL,
    ITM_DMS_KEY             int           NULL,
    WHS_DMS_KEY             int           NULL,
    PFT_CTR_DMS_KEY         int           NULL,

    PCH_ORD_DT_DMS_KEY      int           NULL,  /* Purchase Order Date          */
    PCH_ORD_RCT_DT_DMS_KEY  int           NULL,  /* Receipt Date  — the DEFAULT  */
    CFM_DLY_DT_DMS_KEY      int           NULL,  /* Confirmed Delivery Date      */
    LST_MOD_DT_DMS_KEY      int           NULL,  /* Last Modified Date — AUDIT   */

    PCH_ORD_LIN_AMT         decimal(18,2) NULL,  /* local currency               */
    PCH_ORD_LIN_CAD_AMT     decimal(18,2) NULL,  /* Canadian dollars             */
    PCH_ORD_QTY             decimal(18,3) NULL,
    RCT_QTY                 decimal(18,3) NULL,
    RJT_QTY                 decimal(18,3) NULL,
    UNT_CST_AMT             decimal(18,4) NULL,
    CNY_CD                  nvarchar(4)   NULL,
    RCT_STS_CD              nvarchar(10)  NULL,
    CFM_FLG                 bit           NOT NULL,  /* "confirmed purchase orders" */
    AZ_LST_UPD_TS           datetime2(0)  NULL,
    CONSTRAINT PK_PCH_ORD_RCT_FCT PRIMARY KEY CLUSTERED (PCH_ORD_RCT_FCT_KEY)
);

/* ══════════════════════════════════════════════════════════════════════════
   FACT 3 — FINANCE. A rival fact that also reaches PFT_CTR_DMS, so a metric
   spanning it and the invoice fact produces the multi-fact plan that left the
   graph with no anchor.
   ══════════════════════════════════════════════════════════════════════════ */
CREATE TABLE EMDW_DMART.FNN_FCT (
    FNN_FCT_KEY             bigint        NOT NULL,
    VCH_NO                  nvarchar(20)  NOT NULL,
    VCH_LIN_NO              int           NOT NULL,
    PFT_CTR_DMS_KEY         int           NULL,
    CUS_DMS_KEY             int           NULL,
    ACG_DT_DMS_KEY          int           NULL,  /* Accounting Date — the DEFAULT */
    ENT_DT_DMS_KEY          int           NULL,  /* Entry Date                    */
    DUE_DT_DMS_KEY          int           NULL,  /* Due Date                      */
    LST_MOD_DT_DMS_KEY      int           NULL,  /* Last Modified Date — AUDIT    */
    GL_ACC_CD               nvarchar(20)  NULL,
    GL_ACC_NM               nvarchar(80)  NULL,
    DBT_AMT                 decimal(18,2) NULL,
    CRD_AMT                 decimal(18,2) NULL,
    NET_AMT                 decimal(18,2) NULL,
    NET_CAD_AMT             decimal(18,2) NULL,
    CNY_CD                  nvarchar(4)   NULL,
    AZ_LST_UPD_TS           datetime2(0)  NULL,
    CONSTRAINT PK_FNN_FCT PRIMARY KEY CLUSTERED (FNN_FCT_KEY)
);

/* ══════════════════════════════════════════════════════════════════════════
   FACT 4 — ITEM BALANCE BY PERIOD. Semi-additive month-end snapshot.
   PRD_DMS_KEY is a YYYYMM integer, which is the classification the audit
   flagged as silently wrong for "last 6 months" style windows.
   ══════════════════════════════════════════════════════════════════════════ */
CREATE TABLE EMDW_DMART.ITM_BAL_PRD_FCT (
    ITM_BAL_PRD_FCT_KEY     bigint        NOT NULL,
    ITM_DMS_KEY             int           NULL,
    WHS_DMS_KEY             int           NULL,
    PFT_CTR_DMS_KEY         int           NULL,
    SUP_DMS_KEY             int           NULL,
    PRD_DMS_KEY             int           NULL,  /* 202606 — YYYYMM integer      */
    BAL_DT_DMS_KEY          int           NULL,  /* month-end date — the DEFAULT */
    LST_RCT_DT_DMS_KEY      int           NULL,  /* Last Receipt Date            */
    OH_QTY                  decimal(18,3) NULL,  /* on hand                      */
    ALC_QTY                 decimal(18,3) NULL,  /* allocated                    */
    AVL_QTY                 decimal(18,3) NULL,  /* available                    */
    PCH_QTY                 decimal(18,3) NULL,  /* purchased in period          */
    UNT_CST_AMT             decimal(18,4) NULL,
    BAL_VAL_AMT             decimal(18,2) NULL,
    CONSTRAINT PK_ITM_BAL_PRD_FCT PRIMARY KEY CLUSTERED (ITM_BAL_PRD_FCT_KEY)
);

/*  Foreign keys are declared so schema discovery can find the join candidates
    on its own — that is part of what we want to exercise. They are NOT indexed
    beyond the primary keys, deliberately: the client cannot add indexes, so the
    test model must not quietly be faster than production. */
ALTER TABLE EMDW_DMART.PFT_CTR_DMS ADD CONSTRAINT FK_PFT_CTR_DMS_PC_DVN
    FOREIGN KEY (PC_DVN_DMS_KEY) REFERENCES EMDW_DMART.PC_DVN_DMS (PC_DVN_DMS_KEY);
ALTER TABLE EMDW_DMART.CUS_DMS ADD CONSTRAINT FK_CUS_DMS_CUS_TYP
    FOREIGN KEY (CUS_TYP_DMS_KEY) REFERENCES EMDW_DMART.CUS_TYP_DMS (CUS_TYP_DMS_KEY);
ALTER TABLE EMDW_DMART.CUS_DMS ADD CONSTRAINT FK_CUS_DMS_CUS_SEG
    FOREIGN KEY (CUS_SEG_DMS_KEY) REFERENCES EMDW_DMART.CUS_SEG_DMS (CUS_SEG_DMS_KEY);
ALTER TABLE EMDW_DMART.CUS_DMS ADD CONSTRAINT FK_CUS_DMS_PFT_CTR
    FOREIGN KEY (PFT_CTR_DMS_KEY) REFERENCES EMDW_DMART.PFT_CTR_DMS (PFT_CTR_DMS_KEY);
ALTER TABLE EMDW_DMART.WHS_DMS ADD CONSTRAINT FK_WHS_DMS_PFT_CTR
    FOREIGN KEY (PFT_CTR_DMS_KEY) REFERENCES EMDW_DMART.PFT_CTR_DMS (PFT_CTR_DMS_KEY);
ALTER TABLE EMDW_DMART.PFT_CTR_CUS_DAT ADD CONSTRAINT FK_PFT_CTR_CUS_DAT_PFT_CTR
    FOREIGN KEY (PFT_CTR_DMS_KEY) REFERENCES EMDW_DMART.PFT_CTR_DMS (PFT_CTR_DMS_KEY);
ALTER TABLE EMDW_DMART.PFT_CTR_CUS_DAT ADD CONSTRAINT FK_PFT_CTR_CUS_DAT_CUS
    FOREIGN KEY (CUS_DMS_KEY) REFERENCES EMDW_DMART.CUS_DMS (CUS_DMS_KEY);

ALTER TABLE EMDW_DMART.CUS_ORD_IVC_FCT ADD CONSTRAINT FK_IVC_CUS
    FOREIGN KEY (CUS_DMS_KEY) REFERENCES EMDW_DMART.CUS_DMS (CUS_DMS_KEY);
ALTER TABLE EMDW_DMART.CUS_ORD_IVC_FCT ADD CONSTRAINT FK_IVC_ITM
    FOREIGN KEY (ITM_DMS_KEY) REFERENCES EMDW_DMART.ITM_DMS (ITM_DMS_KEY);
ALTER TABLE EMDW_DMART.CUS_ORD_IVC_FCT ADD CONSTRAINT FK_IVC_WHS
    FOREIGN KEY (WHS_DMS_KEY) REFERENCES EMDW_DMART.WHS_DMS (WHS_DMS_KEY);
ALTER TABLE EMDW_DMART.CUS_ORD_IVC_FCT ADD CONSTRAINT FK_IVC_PFT_CTR
    FOREIGN KEY (PFT_CTR_DMS_KEY) REFERENCES EMDW_DMART.PFT_CTR_DMS (PFT_CTR_DMS_KEY);
ALTER TABLE EMDW_DMART.CUS_ORD_IVC_FCT ADD CONSTRAINT FK_IVC_INVOICE_DATE
    FOREIGN KEY (CUS_IVC_DT_DMS_KEY) REFERENCES EMDW_DMART.DT_DMS (DT_DMS_KEY);
ALTER TABLE EMDW_DMART.CUS_ORD_IVC_FCT ADD CONSTRAINT FK_IVC_ORDER_DATE
    FOREIGN KEY (CUS_ORD_DT_DMS_KEY) REFERENCES EMDW_DMART.DT_DMS (DT_DMS_KEY);
ALTER TABLE EMDW_DMART.CUS_ORD_IVC_FCT ADD CONSTRAINT FK_IVC_CFM_DLY_DATE
    FOREIGN KEY (CFM_DLY_DT_DMS_KEY) REFERENCES EMDW_DMART.DT_DMS (DT_DMS_KEY);
ALTER TABLE EMDW_DMART.CUS_ORD_IVC_FCT ADD CONSTRAINT FK_IVC_RQS_DLY_DATE
    FOREIGN KEY (RQS_DLY_DT_DMS_KEY) REFERENCES EMDW_DMART.DT_DMS (DT_DMS_KEY);
ALTER TABLE EMDW_DMART.CUS_ORD_IVC_FCT ADD CONSTRAINT FK_IVC_PLN_DLY_DATE
    FOREIGN KEY (PLN_DLY_DT_DMS_KEY) REFERENCES EMDW_DMART.DT_DMS (DT_DMS_KEY);
ALTER TABLE EMDW_DMART.CUS_ORD_IVC_FCT ADD CONSTRAINT FK_IVC_CNL_ORD_DATE
    FOREIGN KEY (CNL_ORD_DT_DMS_KEY) REFERENCES EMDW_DMART.DT_DMS (DT_DMS_KEY);
ALTER TABLE EMDW_DMART.CUS_ORD_IVC_FCT ADD CONSTRAINT FK_IVC_DUE_DATE
    FOREIGN KEY (DUE_DT_DMS_KEY) REFERENCES EMDW_DMART.DT_DMS (DT_DMS_KEY);
ALTER TABLE EMDW_DMART.CUS_ORD_IVC_FCT ADD CONSTRAINT FK_IVC_LST_MOD_DATE
    FOREIGN KEY (LST_MOD_DT_DMS_KEY) REFERENCES EMDW_DMART.DT_DMS (DT_DMS_KEY);

ALTER TABLE EMDW_DMART.PCH_ORD_RCT_FCT ADD CONSTRAINT FK_PCH_SUP
    FOREIGN KEY (SUP_DMS_KEY) REFERENCES EMDW_DMART.SUP_DMS (SUP_DMS_KEY);
ALTER TABLE EMDW_DMART.PCH_ORD_RCT_FCT ADD CONSTRAINT FK_PCH_ITM
    FOREIGN KEY (ITM_DMS_KEY) REFERENCES EMDW_DMART.ITM_DMS (ITM_DMS_KEY);
ALTER TABLE EMDW_DMART.PCH_ORD_RCT_FCT ADD CONSTRAINT FK_PCH_WHS
    FOREIGN KEY (WHS_DMS_KEY) REFERENCES EMDW_DMART.WHS_DMS (WHS_DMS_KEY);
ALTER TABLE EMDW_DMART.PCH_ORD_RCT_FCT ADD CONSTRAINT FK_PCH_PFT_CTR
    FOREIGN KEY (PFT_CTR_DMS_KEY) REFERENCES EMDW_DMART.PFT_CTR_DMS (PFT_CTR_DMS_KEY);
ALTER TABLE EMDW_DMART.PCH_ORD_RCT_FCT ADD CONSTRAINT FK_PCH_ORDER_DATE
    FOREIGN KEY (PCH_ORD_DT_DMS_KEY) REFERENCES EMDW_DMART.DT_DMS (DT_DMS_KEY);
ALTER TABLE EMDW_DMART.PCH_ORD_RCT_FCT ADD CONSTRAINT FK_PCH_RECEIPT_DATE
    FOREIGN KEY (PCH_ORD_RCT_DT_DMS_KEY) REFERENCES EMDW_DMART.DT_DMS (DT_DMS_KEY);

ALTER TABLE EMDW_DMART.FNN_FCT ADD CONSTRAINT FK_FNN_PFT_CTR
    FOREIGN KEY (PFT_CTR_DMS_KEY) REFERENCES EMDW_DMART.PFT_CTR_DMS (PFT_CTR_DMS_KEY);
ALTER TABLE EMDW_DMART.FNN_FCT ADD CONSTRAINT FK_FNN_CUS
    FOREIGN KEY (CUS_DMS_KEY) REFERENCES EMDW_DMART.CUS_DMS (CUS_DMS_KEY);
ALTER TABLE EMDW_DMART.FNN_FCT ADD CONSTRAINT FK_FNN_ACG_DATE
    FOREIGN KEY (ACG_DT_DMS_KEY) REFERENCES EMDW_DMART.DT_DMS (DT_DMS_KEY);

ALTER TABLE EMDW_DMART.ITM_BAL_PRD_FCT ADD CONSTRAINT FK_BAL_ITM
    FOREIGN KEY (ITM_DMS_KEY) REFERENCES EMDW_DMART.ITM_DMS (ITM_DMS_KEY);
ALTER TABLE EMDW_DMART.ITM_BAL_PRD_FCT ADD CONSTRAINT FK_BAL_WHS
    FOREIGN KEY (WHS_DMS_KEY) REFERENCES EMDW_DMART.WHS_DMS (WHS_DMS_KEY);
ALTER TABLE EMDW_DMART.ITM_BAL_PRD_FCT ADD CONSTRAINT FK_BAL_PFT_CTR
    FOREIGN KEY (PFT_CTR_DMS_KEY) REFERENCES EMDW_DMART.PFT_CTR_DMS (PFT_CTR_DMS_KEY);
ALTER TABLE EMDW_DMART.ITM_BAL_PRD_FCT ADD CONSTRAINT FK_BAL_DATE
    FOREIGN KEY (BAL_DT_DMS_KEY) REFERENCES EMDW_DMART.DT_DMS (DT_DMS_KEY);

SELECT 'EMDW_DMART model created' AS STATUS,
       COUNT(*) AS TABLE_COUNT
FROM sys.tables t
JOIN sys.schemas s ON s.schema_id = t.schema_id
WHERE s.name = 'EMDW_DMART';
