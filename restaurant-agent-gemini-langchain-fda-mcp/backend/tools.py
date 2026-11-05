from datetime import date, timedelta


from langchain_core.tools import tool

# ============================================================
# TOOL 1: Search Restaurants by Cuisine
# ============================================================

@tool
def search_restaurants_by_cuisine(cuisine_type: str) -> list:
    """
    Search for restaurants by cuisine type.
    Supported cuisines: Indian, Italian, Chinese, Mexican.
    """

    restaurants = {
        "indian": [
            {"name": "Spice Palace", "rating": 4.5, "price": "₹₹"},
            {"name": "Curry House", "rating": 4.3, "price": "₹₹"},
            {"name": "Taj Flavors", "rating": 4.7, "price": "₹₹₹"}
        ],
        "italian": [
            {"name": "Pizza Bella", "rating": 4.4, "price": "₹₹"},
            {"name": "Pasta Dreams", "rating": 4.6, "price": "₹₹₹"},
            {"name": "Roma Kitchen", "rating": 4.2, "price": "₹₹"}
        ],
        "chinese": [
            {"name": "Dragon House", "rating": 4.5, "price": "₹₹"},
            {"name": "Ming's Kitchen", "rating": 4.3, "price": "₹₹"},
            {"name": "Golden Wok", "rating": 4.8, "price": "₹₹₹"}
        ],
        "mexican": [
            {"name": "Taco Fiesta", "rating": 4.4, "price": "₹₹"},
            {"name": "Salsa Nights", "rating": 4.6, "price": "₹₹₹"},
            {"name": "Burrito House", "rating": 4.2, "price": "₹₹"}
        ]
    }

    cuisine_lower = cuisine_type.lower().strip()

    return restaurants.get(cuisine_lower, [])


# ============================================================
# TOOL 2: Get Restaurant Details
# ============================================================

@tool
def get_restaurant_details(restaurant_name: str) -> dict:
    """
    Get detailed information about a restaurant,
    including address, phone, hours and specialties.
    """

    details = {
        "Spice Palace": {
            "cuisine": "Indian",
            "rating": 4.5,
            "address": "123 Food Street, City Center",
            "phone": "9876543210",
            "hours": "11 AM - 11 PM",
            "speciality": "Butter Chicken, Biryani"
        },

        "Pizza Bella": {
            "cuisine": "Italian",
            "rating": 4.4,
            "address": "456 Dine Avenue, Downtown",
            "phone": "9123456789",
            "hours": "12 PM - 10:30 PM",
            "speciality": "Margherita Pizza, Tiramisu"
        },

        "Dragon House": {
            "cuisine": "Chinese",
            "rating": 4.5,
            "address": "789 Noodle Lane, Market Area",
            "phone": "9111222333",
            "hours": "11:30 AM - 11 PM",
            "speciality": "Hakka Noodles, Fried Rice"
        },

        "Taco Fiesta": {
            "cuisine": "Mexican",
            "rating": 4.4,
            "address": "321 Spice Row, Hip District",
            "phone": "9222333444",
            "hours": "1 PM - 12 AM",
            "speciality": "Fish Tacos, Guacamole"
        },

        "Curry House": {
            "cuisine": "Indian",
            "rating": 4.3,
            "address": "555 Flavor Street, Old Town",
            "phone": "9333444555",
            "hours": "11 AM - 10 PM",
            "speciality": "Paneer Tikka, Dal Makhani"
        },

        "Taj Flavors": {
            "cuisine": "Indian",
            "rating": 4.7,
            "address": "999 Taste Lane, Premium Zone",
            "phone": "9444555666",
            "hours": "12 PM - 11 PM",
            "speciality": "Biryani, Tandoori Chicken"
        },

        "Golden Wok": {
            "cuisine": "Chinese",
            "rating": 4.8,
            "address": "777 Wok Street, Central Market",
            "phone": "9555666777",
            "hours": "11 AM - 11 PM",
            "speciality": "Szechuan Noodles, Dim Sum"
        },

        "Pasta Dreams": {
            "cuisine": "Italian",
            "rating": 4.6,
            "address": "12 Olive Road, Riverside",
            "phone": "9666777888",
            "hours": "12 PM - 11 PM",
            "speciality": "Truffle Pasta, Lasagna"
        },

        "Roma Kitchen": {
            "cuisine": "Italian",
            "rating": 4.2,
            "address": "34 Piazza Lane, Old Town",
            "phone": "9777888999",
            "hours": "11:30 AM - 10:30 PM",
            "speciality": "Risotto, Wood-fired Pizza"
        },

        "Ming's Kitchen": {
            "cuisine": "Chinese",
            "rating": 4.3,
            "address": "56 Lantern Street, Chinatown",
            "phone": "9888999000",
            "hours": "12 PM - 11 PM",
            "speciality": "Manchurian, Hot and Sour Soup"
        },

        "Salsa Nights": {
            "cuisine": "Mexican",
            "rating": 4.6,
            "address": "78 Fiesta Avenue, Hip District",
            "phone": "9999000111",
            "hours": "2 PM - 12:30 AM",
            "speciality": "Enchiladas, Churros"
        },

        "Burrito House": {
            "cuisine": "Mexican",
            "rating": 4.2,
            "address": "90 Cantina Road, Market Area",
            "phone": "9000111222",
            "hours": "11 AM - 10 PM",
            "speciality": "Loaded Burritos, Nachos"
        }
    }

    return details.get(
        restaurant_name.strip(),
        {"error": f"Restaurant '{restaurant_name}' not found"}
    )


# ============================================================
# TOOL 3: Check Reservation Availability
# ============================================================

# Bookings are taken for the next BOOKING_DAYS days, starting tomorrow.
BOOKING_DAYS = 7

# Each restaurant: the weekday it is closed (None = open every day) and
# its time slots (24h "HH:MM") with the seats still free in each slot.
# On Saturdays and Sundays, dinner slots (19:00 and later) have half the seats.
RESERVATION_SLOTS = {
    "Spice Palace": ("Monday", {"12:00": 8, "13:30": 4, "19:00": 2, "20:30": 6}),
    "Curry House": ("Tuesday", {"12:00": 6, "13:30": 0, "19:00": 4, "20:30": 2}),
    "Taj Flavors": ("Monday", {"12:30": 4, "14:00": 2, "19:30": 0, "21:00": 6}),
    "Pizza Bella": ("Wednesday", {"12:30": 6, "14:00": 6, "19:00": 0, "20:30": 4}),
    "Pasta Dreams": ("Monday", {"12:30": 2, "14:00": 4, "19:30": 6, "21:00": 0}),
    "Roma Kitchen": ("Thursday", {"12:00": 8, "13:30": 2, "19:00": 4, "20:30": 0}),
    "Dragon House": ("Tuesday", {"12:00": 4, "13:30": 6, "19:00": 2, "21:00": 8}),
    "Ming's Kitchen": (None, {"12:30": 6, "14:00": 4, "19:30": 4, "21:00": 2}),
    "Golden Wok": ("Wednesday", {"12:00": 0, "13:30": 2, "19:00": 6, "20:30": 4}),
    "Taco Fiesta": (None, {"13:00": 8, "15:00": 6, "19:00": 4, "22:00": 6}),
    "Salsa Nights": ("Monday", {"14:00": 2, "17:00": 4, "20:00": 0, "22:30": 0}),
    "Burrito House": ("Sunday", {"12:00": 4, "14:00": 0, "18:30": 2, "20:00": 2}),
}


def booking_dates() -> list[str]:
    """The dates (YYYY-MM-DD) open for booking."""
    today = date.today()
    return [(today + timedelta(days=i)).isoformat() for i in range(1, BOOKING_DAYS + 1)]


def is_closed(restaurant_name: str, day: str) -> bool:
    closed_on, _ = RESERVATION_SLOTS[restaurant_name]
    return date.fromisoformat(day).strftime("%A") == closed_on


def seats_left(restaurant_name: str, day: str) -> dict[str, int]:
    """Free seats per time slot on `day` (empty if the restaurant is closed)."""
    if is_closed(restaurant_name, day):
        return {}
    _, slots = RESERVATION_SLOTS[restaurant_name]
    weekend = date.fromisoformat(day).weekday() >= 5
    return {
        t: seats // 2 if weekend and t >= "19:00" else seats
        for t, seats in slots.items()
    }


def free_times(restaurant_name: str, day: str, party_size: int) -> list[str]:
    """The time slots on `day` that can still seat `party_size` people."""
    return [t for t, seats in seats_left(restaurant_name, day).items() if seats >= party_size]


@tool
def check_reservation_availability(
    restaurant_name: str,
    date: str,
    party_size: int,
    time: str = ""
) -> str:
    """
    Check reservation availability at a restaurant for a date (YYYY-MM-DD)
    and number of people. Give a time (24h HH:MM) to check one slot, or
    leave it empty to list every time slot that can seat the party that day.
    Bookings are open for the next 7 days, starting tomorrow.
    Only call this once the user has told you both the date and the number
    of people; if either is missing, ask the user instead of guessing.
    """

    if restaurant_name not in RESERVATION_SLOTS:
        return f"Restaurant '{restaurant_name}' not found."

    dates = booking_dates()
    if date not in dates:
        return (
            f"No bookings for {date}: reservations are open "
            f"from {dates[0]} to {dates[-1]}."
        )

    if is_closed(restaurant_name, date):
        return f"✗ {restaurant_name} is closed on {date}."

    if time:
        seats = seats_left(restaurant_name, date)
        if time not in seats:
            return (
                f"✗ {restaurant_name} has no {time} slot. "
                f"Times: {', '.join(seats)}"
            )
        if seats[time] >= party_size:
            return (
                f"✓ {restaurant_name} has availability "
                f"for {party_size} people on {date} at {time}"
            )
        return (
            f"✗ {restaurant_name} is fully booked "
            f"for {party_size} people on {date} at {time}"
        )

    times = free_times(restaurant_name, date, party_size)
    if times:
        return (
            f"✓ {restaurant_name} has availability "
            f"for {party_size} people on {date} at {', '.join(times)}"
        )

    return (
        f"✗ {restaurant_name} is fully booked "
        f"for {party_size} people on {date}"
    )


tools = [search_restaurants_by_cuisine, get_restaurant_details, check_reservation_availability]
