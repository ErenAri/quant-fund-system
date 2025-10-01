"""
Alpaca broker integration for live/paper trading.

Handles:
- Account info
- Position management
- Order execution
- Market data (backup to primary feed)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime

import pandas as pd
import requests
from quantfund.utils.log import get_logger

logger = get_logger(__name__)


@dataclass
class Position:
    symbol: str
    qty: float
    market_value: float
    avg_entry_price: float
    current_price: float
    unrealized_pl: float
    unrealized_pl_pct: float


@dataclass
class Order:
    symbol: str
    qty: float
    side: str  # "buy" or "sell"
    order_type: str  # "market" or "limit"
    time_in_force: str  # "day", "gtc", etc
    limit_price: Optional[float] = None


class AlpacaBroker:
    """Alpaca API client for trading."""

    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None, paper: bool = True):
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")

        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API credentials not found. Set ALPACA_API_KEY and ALPACA_SECRET_KEY env vars.")

        # Standard individual account API endpoints
        if paper:
            self.base_url = "https://paper-api.alpaca.markets"
            self.data_url = "https://data.alpaca.markets"
        else:
            self.base_url = "https://api.alpaca.markets"
            self.data_url = "https://data.alpaca.markets"

        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
        }

        logger.info(f"Initialized Alpaca broker (paper={paper}, endpoint={self.base_url})")

    def _request(self, method: str, endpoint: str, **kwargs) -> dict:
        """Make authenticated request to Alpaca API."""
        url = f"{self.base_url}{endpoint}"
        response = requests.request(method, url, headers=self.headers, **kwargs)
        response.raise_for_status()
        return response.json()

    def get_account(self) -> dict:
        """Get account information."""
        return self._request("GET", "/v2/account")

    def get_positions(self) -> List[Position]:
        """Get current positions."""
        data = self._request("GET", "/v2/positions")
        positions = []
        for p in data:
            positions.append(Position(
                symbol=p["symbol"],
                qty=float(p["qty"]),
                market_value=float(p["market_value"]),
                avg_entry_price=float(p["avg_entry_price"]),
                current_price=float(p["current_price"]),
                unrealized_pl=float(p["unrealized_pl"]),
                unrealized_pl_pct=float(p["unrealized_plpc"]),
            ))
        return positions

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for specific symbol."""
        try:
            p = self._request("GET", f"/v2/positions/{symbol}")
            return Position(
                symbol=p["symbol"],
                qty=float(p["qty"]),
                market_value=float(p["market_value"]),
                avg_entry_price=float(p["avg_entry_price"]),
                current_price=float(p["current_price"]),
                unrealized_pl=float(p["unrealized_pl"]),
                unrealized_pl_pct=float(p["unrealized_plpc"]),
            )
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                return None
            raise

    def close_all_positions(self) -> dict:
        """Close all open positions."""
        return self._request("DELETE", "/v2/positions")

    def submit_order(self, order: Order) -> dict:
        """Submit an order."""
        payload = {
            "symbol": order.symbol,
            "qty": abs(order.qty),
            "side": order.side,
            "type": order.order_type,
            "time_in_force": order.time_in_force,
        }

        if order.limit_price:
            payload["limit_price"] = order.limit_price

        logger.info(f"Submitting order: {order.side} {order.qty} {order.symbol} @ {order.order_type}")
        return self._request("POST", "/v2/orders", json=payload)

    def get_orders(self, status: str = "open") -> List[dict]:
        """Get orders by status."""
        return self._request("GET", "/v2/orders", params={"status": status})

    def cancel_all_orders(self) -> List[dict]:
        """Cancel all open orders."""
        return self._request("DELETE", "/v2/orders")

    def get_bars(self, symbols: List[str], timeframe: str = "1Day", limit: int = 100) -> pd.DataFrame:
        """
        Get historical bars (backup data source).

        timeframe: "1Min", "5Min", "15Min", "1Hour", "1Day"
        """
        symbols_str = ",".join(symbols)
        data = self._request("GET", "/v2/stocks/bars", params={
            "symbols": symbols_str,
            "timeframe": timeframe,
            "limit": limit,
        })

        dfs = []
        for symbol, bars in data["bars"].items():
            df = pd.DataFrame(bars)
            df["symbol"] = symbol
            dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)
        combined["t"] = pd.to_datetime(combined["t"])
        return combined

    def get_latest_quotes(self, symbols: List[str]) -> Dict[str, float]:
        """Get latest quotes for symbols."""
        symbols_str = ",".join(symbols)
        # Use data API endpoint (same for broker and standard)
        data_headers = self.headers.copy()
        url = f"https://data.alpaca.markets/v2/stocks/quotes/latest?symbols={symbols_str}"
        response = requests.get(url, headers=data_headers)
        response.raise_for_status()
        data = response.json()

        quotes = {}
        for symbol, quote in data["quotes"].items():
            # Use mid price (bid + ask) / 2
            bid = float(quote.get("bp", 0))
            ask = float(quote.get("ap", 0))
            quotes[symbol] = (bid + ask) / 2 if bid and ask else float(quote.get("p", 0))

        return quotes


__all__ = ["AlpacaBroker", "Position", "Order"]
