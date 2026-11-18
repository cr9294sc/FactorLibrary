// 按市场隔离的前端应用状态中心

export const VALID_MARKETS = ["futures", "stocks"];
export const VALID_PAGES = ["data", "lab", "compare", "mining", "registry"];

function defaultMarketState() {
  return {
    context: null,
    extensions: [],
    featureSets: [],
    activeFeatureSetId: null,
    klineState: {
      symbol: null,
      basket: [],
      sector: "全部",
      ind: { MA5: true, MA20: true, MA60: false, BOLL: false },
      mode: "k",
      range: 120,
      adj: "hfq",
      sub: "vol",
      query: "",
      data: null,
    },
    lab: {
      catTab: "字段",
      catQuery: "",
      factorType: "expression",
      postprocess: [],
      probeResult: null,
      catalogCache: [],
    },
    compare: {
      mode: "single", // single | ab
      selectedIds: [],
      workflowResult: null,
      abResult: null,
    },
    mining: {
      kind: "dl", // gp | dl | ml
      lastRecipe: null,
      runs: [],
      openLogRunId: null,
    },
    registry: {
      view: "cand", // cand | ledger
      sourceFilter: "",
      runFilter: "",
      minIcir: 0,
      factors: [],
      selectedFactorIds: new Set(),
      auditMap: new Map(),
      detailFactor: null,
    },
  };
}

export const store = {
  market: "futures",
  page: "data",
  byMarket: {
    futures: defaultMarketState(),
    stocks: defaultMarketState(),
  },
  pollTimer: null,
};

export function ms(market = store.market) {
  return store.byMarket[market];
}
