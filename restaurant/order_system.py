from __future__ import annotations

from dataclasses import asdict, dataclass


MENU = (
    "Chicken Burger",
    "Margherita Pizza",
    "Pasta Alfredo",
    "Biryani Bowl",
    "Club Sandwich",
    "Coffee",
    "Fresh Juice",
)


@dataclass
class RestaurantOrder:
    order_id: str
    table_id: int
    item: str
    quantity: int
    status: str = "PLACED"

    def to_dict(self) -> dict:
        return asdict(self)


class OrderSystem:
    """Order-domain layer shared by the simulator and Tkinter dashboard."""

    def __init__(self) -> None:
        self.sequence = 0
        self.current_order: RestaurantOrder | None = None
        self.history: list[RestaurantOrder] = []

    def create_order(
        self,
        rng,
        table_id: int,
        item: str | None = None,
        quantity: int | None = None,
    ) -> RestaurantOrder:
        self.sequence += 1
        selected_item = item or str(rng.choice(MENU))
        selected_quantity = (
            int(rng.integers(1, 4))
            if quantity is None
            else int(max(1, min(3, quantity)))
        )
        order = RestaurantOrder(
            order_id=f"ORD-{self.sequence:04d}",
            table_id=int(table_id),
            item=selected_item,
            quantity=selected_quantity,
        )
        self.current_order = order
        self.history.append(order)
        return order

    def mark_preparing(self) -> None:
        if self.current_order is not None:
            self.current_order.status = "PREPARING"

    def mark_on_robot(self) -> None:
        if self.current_order is not None:
            self.current_order.status = "ON_ROBOT"

    def mark_delivered(self) -> None:
        if self.current_order is not None:
            self.current_order.status = "DELIVERED"

    def mark_failed(self) -> None:
        if self.current_order is not None:
            self.current_order.status = "FAILED"
