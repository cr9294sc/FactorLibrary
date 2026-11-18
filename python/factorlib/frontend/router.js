// 零构建 Hash 路由器 (#/{market}/{page})

import { VALID_MARKETS, VALID_PAGES, store } from "./state.js";

export function parseHash(hashStr = window.location.hash) {
  const m = (hashStr || "").match(/^#\/(futures|stocks)\/(\w+)/);
  if (!m) {
    return { market: store.market, page: store.page };
  }
  const market = VALID_MARKETS.includes(m[1]) ? m[1] : "futures";
  const page = VALID_PAGES.includes(m[2]) ? m[2] : "data";
  return { market, page };
}

export function setHash(market, page) {
  const next = `#/${market}/${page}`;
  if (window.location.hash !== next) {
    window.history.replaceState(null, "", next);
  }
}
