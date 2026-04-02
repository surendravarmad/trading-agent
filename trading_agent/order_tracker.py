"""
Order Tracker
==============
Fetches and monitors order statuses from the Alpaca API.
Provides visibility into pending, filled, partially filled,
cancelled, and rejected orders.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    NEW = "new"
    ACCEPTED = "accepted"
    PENDING_NEW = "pending_new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    DONE_FOR_DAY = "done_for_day"
    CANCELED = "canceled"
    EXPIRED = "expired"
    REPLACED = "replaced"
    REJECTED = "rejected"
    HELD = "held"
    UNKNOWN = "unknown"


@dataclass
class OrderRecord:
    """Tracked order from Alpaca."""
    order_id: str
    status: OrderStatus
    symbol: str
    side: str
    order_type: str
    order_class: str
    qty: str
    filled_qty: str
    limit_price: Optional[str]
    filled_avg_price: Optional[str]
    created_at: str
    updated_at: str
    legs: List[Dict]
    raw: Dict


class OrderTracker:
    """
    Queries Alpaca's /v2/orders endpoint to track order lifecycle:
      - Open orders (pending fill)
      - Recently filled orders
      - Failed/rejected orders
    """

    def __init__(self, api_key: str, secret_key: str,
                 base_url: str = "https://paper-api.alpaca.markets/v2"):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url

    def _headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Accept": "application/json",
        }

    # ------------------------------------------------------------------
    # Fetch orders
    # ------------------------------------------------------------------

    def fetch_orders(self, status: str = "all",
                     limit: int = 50,
                     direction: str = "desc") -> List[OrderRecord]:
        """
        GET /v2/orders — fetch orders filtered by status.

        Args:
            status: "open", "closed", or "all"
            limit:  max number of orders to return (default 50)
            direction: "asc" or "desc" by created_at
        """
        url = f"{self.base_url}/orders"
        params = {
            "status": status,
            "limit": limit,
            "direction": direction,
        }
        try:
            resp = requests.get(url, headers=self._headers(),
                                params=params, timeout=10)
            resp.raise_for_status()
            orders_data = resp.json()

            records = []
            for o in orders_data:
                record = self._parse_order(o)
                records.append(record)

            logger.info("Fetched %d orders (status=%s)", len(records), status)
            return records

        except requests.RequestException as exc:
            logger.error("Failed to fetch orders: %s", exc)
            return []

    def fetch_open_orders(self) -> List[OrderRecord]:
        """Convenience: fetch only open/pending orders."""
        return self.fetch_orders(status="open")

    def fetch_recent_fills(self, limit: int = 20) -> List[OrderRecord]:
        """Convenience: fetch recently closed orders (filled, cancelled, etc)."""
        all_closed = self.fetch_orders(status="closed", limit=limit)
        return [o for o in all_closed if o.status == OrderStatus.FILLED]

    def get_order_by_id(self, order_id: str) -> Optional[OrderRecord]:
        """GET /v2/orders/{order_id} — fetch a specific order."""
        url = f"{self.base_url}/orders/{order_id}"
        try:
            resp = requests.get(url, headers=self._headers(), timeout=10)
            resp.raise_for_status()
            return self._parse_order(resp.json())
        except requests.RequestException as exc:
            logger.error("Failed to fetch order %s: %s", order_id, exc)
            return None

    # ------------------------------------------------------------------
    # Cancel orders
    # ------------------------------------------------------------------

    def cancel_order(self, order_id: str) -> bool:
        """DELETE /v2/orders/{order_id} — cancel a specific open order."""
        url = f"{self.base_url}/orders/{order_id}"
        try:
            resp = requests.delete(url, headers=self._headers(), timeout=10)
            resp.raise_for_status()
            logger.info("Cancelled order %s", order_id)
            return True
        except requests.RequestException as exc:
            logger.error("Failed to cancel order %s: %s", order_id, exc)
            return False

    def cancel_all_orders(self) -> bool:
        """DELETE /v2/orders — cancel all open orders."""
        url = f"{self.base_url}/orders"
        try:
            resp = requests.delete(url, headers=self._headers(), timeout=10)
            resp.raise_for_status()
            logger.info("Cancelled all open orders")
            return True
        except requests.RequestException as exc:
            logger.error("Failed to cancel all orders: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def summarize_orders(self, orders: List[OrderRecord]) -> Dict:
        """Produce a summary of order statuses."""
        by_status = {}
        for o in orders:
            key = o.status.value
            by_status[key] = by_status.get(key, 0) + 1

        return {
            "total": len(orders),
            "by_status": by_status,
            "open": sum(1 for o in orders if o.status in (
                OrderStatus.NEW, OrderStatus.ACCEPTED,
                OrderStatus.PENDING_NEW, OrderStatus.PARTIALLY_FILLED,
                OrderStatus.HELD)),
            "filled": sum(1 for o in orders if o.status == OrderStatus.FILLED),
            "failed": sum(1 for o in orders if o.status in (
                OrderStatus.REJECTED, OrderStatus.CANCELED,
                OrderStatus.EXPIRED)),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _parse_order(self, data: Dict) -> OrderRecord:
        status_str = data.get("status", "unknown")
        try:
            status = OrderStatus(status_str)
        except ValueError:
            status = OrderStatus.UNKNOWN

        legs_data = data.get("legs", []) or []
        legs = []
        for leg in legs_data:
            legs.append({
                "symbol": leg.get("symbol", ""),
                "side": leg.get("side", ""),
                "qty": leg.get("qty", ""),
                "filled_qty": leg.get("filled_qty", ""),
                "status": leg.get("status", ""),
            })

        return OrderRecord(
            order_id=data.get("id", ""),
            status=status,
            symbol=data.get("symbol", ""),
            side=data.get("side", ""),
            order_type=data.get("type", ""),
            order_class=data.get("order_class", ""),
            qty=data.get("qty", "0"),
            filled_qty=data.get("filled_qty", "0"),
            limit_price=data.get("limit_price"),
            filled_avg_price=data.get("filled_avg_price"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            legs=legs,
            raw=data,
        )
