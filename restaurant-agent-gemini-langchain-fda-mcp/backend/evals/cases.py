"""Evaluation cases for POST /chat.

Each case is one or more turns sent in order; `state` and history carry over
between turns like the frontend does. A turn's `expect` has a `jev` part
and/or a `gemini` part (see run_evals.py for the checks). Dates are built
from tools.booking_dates(), so the cases stay valid as the window moves.

Jev checks:    choice, next_question, next_choices, next_choices_lack,
               tools, no_tools, tool_output_has, reply_has
Gemini checks: tools, tools_any, no_tools, tool_output_has, reply_has,
               reply_lacks, judge

The `fda` cases need the FDA menu rating MCP server (FDA_MCP_URL); without
it Gemini has no rating tools and they fail.
"""

from datetime import date

from flow import date_label
from tools import booking_dates


def _first(pred) -> str:
    return next(d for d in booking_dates() if pred(date.fromisoformat(d)))


def _weekday(*closed: int) -> str:
    """First Mon-Fri booking date that is not one of the `closed` weekdays (0 = Monday)."""
    return _first(lambda d: d.weekday() < 5 and d.weekday() not in closed)


def _long(day: str) -> str:
    """'2026-11-02' -> 'Monday 02 November'."""
    return date.fromisoformat(day).strftime("%A %d %B")


def build_cases() -> list[dict]:
    tomorrow = booking_dates()[0]
    monday = _first(lambda d: d.weekday() == 0)
    saturday = _first(lambda d: d.weekday() == 5)
    spice_day = _weekday(0)  # Spice Palace is closed on Mondays
    salsa_day = _weekday(0)
    taco_day = _weekday()
    taj_day = _weekday(0)

    italian = {"asking": "restaurant", "cuisine": "Italian"}
    taco = {"cuisine": "Mexican", "restaurant": "Taco Fiesta"}

    return [
        # ---------------- Guided flow (clicking the buttons) ----------------
        {
            "id": "F1", "category": "flow",
            "name": "Full booking: Indian -> Spice Palace -> weekday -> 2 -> 7 PM",
            "turns": [
                {"message": "Indian", "expect": {"jev": {
                    "choice": "indian", "next_question": "restaurant",
                    "tools": [{"name": "search_restaurants_by_cuisine", "args": {"cuisine_type": "Indian"}}],
                }}},
                {"message": "Spice Palace", "expect": {"jev": {
                    "choice": "Spice Palace", "next_question": "date",
                    "next_choices_lack": [date_label(monday)],  # closed on Mondays
                }}},
                {"message": date_label(spice_day), "expect": {
                    "jev": {"choice": spice_day, "next_question": "party", "no_tools": True},
                    # Party size is not known yet: Gemini must not guess it.
                    "gemini": {"no_tools": ["check_reservation_availability"],
                               "judge": "Asks how many people are in the party."},
                }},
                {"message": "2 people", "expect": {
                    "jev": {
                        "choice": "2", "next_question": "time",
                        "next_choices": ["12:00 PM", "1:30 PM", "7:00 PM", "8:30 PM"],
                        "tools": [{"name": "check_reservation_availability",
                                   "args": {"restaurant_name": "Spice Palace", "date": spice_day, "party_size": 2}}],
                    },
                    "gemini": {"tools": [{"name": "check_reservation_availability",
                                          "args": {"restaurant_name": "Spice Palace", "date": spice_day, "party_size": 2}}]},
                }},
                {"message": "7:00 PM", "expect": {
                    "jev": {
                        "choice": "19:00", "next_question": "cuisine",
                        "tools": [{"name": "check_reservation_availability", "args": {"time": "19:00"}}],
                        "tool_output_has": "has availability",
                    },
                    "gemini": {
                        "tools": [{"name": "check_reservation_availability",
                                   "args": {"restaurant_name": "Spice Palace", "date": spice_day,
                                            "party_size": 2, "time": "19:00"}}],
                        "judge": "Says a table for 2 is available at 7 PM (19:00).",
                    },
                }},
            ],
        },
        {
            "id": "F2", "category": "flow",
            "name": "Party of 6 at Salsa Nights on a weekday: no slot fits, start over",
            "state": {"asking": "party", "cuisine": "Mexican", "restaurant": "Salsa Nights", "date": salsa_day},
            "turns": [{"message": "6 people", "expect": {"jev": {
                "choice": "6", "next_question": "cuisine",
                "tool_output_has": "fully booked", "reply_has": ["fully booked"],
            }}}],
        },
        {
            "id": "F3", "category": "flow",
            "name": "Start over from the restaurant question",
            "state": italian,
            "turns": [{"message": "None of these - start over", "expect": {"jev": {
                "choice": "restart", "next_question": "cuisine", "no_tools": True,
            }}}],
        },
        {
            "id": "F4", "category": "flow",
            "name": "Saturday dinner has half the seats: 7 PM (1 seat) not offered for 2",
            "state": {"asking": "party", "cuisine": "Indian", "restaurant": "Spice Palace", "date": saturday},
            "turns": [{"message": "2 people", "expect": {"jev": {
                "choice": "2", "next_question": "time",
                "next_choices": ["12:00 PM", "1:30 PM", "8:30 PM"],
            }}}],
        },

        # ---------------- Jev: understanding typed answers ----------------
        {"id": "B1", "category": "jev", "name": "Food -> cuisine",
         "turns": [{"message": "I'm in the mood for pasta",
                    "expect": {"jev": {"choice": "italian", "next_question": "restaurant"}}}]},
        {"id": "B2", "category": "jev", "name": "Dish -> cuisine",
         "turns": [{"message": "something with tacos",
                    "expect": {"jev": {"choice": "mexican", "next_question": "restaurant"}}}]},
        {"id": "B3", "category": "jev", "name": "Unsupported cuisine stays on the question",
         "turns": [{"message": "Japanese please",
                    "expect": {"jev": {"choice": "other", "next_question": "cuisine", "no_tools": True}}}]},
        {"id": "B4", "category": "jev", "name": "'Highest rated' picks Pasta Dreams (4.6)",
         "state": italian,
         "turns": [{"message": "the highest rated one",
                    "expect": {"jev": {"choice": "Pasta Dreams", "next_question": "date"}}}]},
        {"id": "B5", "category": "jev", "name": "Restaurant from another cuisine is not a choice",
         "state": italian,
         "turns": [{"message": "Dragon House", "expect": {"jev": {"choice": "other", "next_question": "restaurant"}}}]},
        {"id": "B6", "category": "jev", "name": "Dislikes options -> restart",
         "state": italian,
         "turns": [{"message": "meh, none of these", "expect": {"jev": {"choice": "restart", "next_question": "cuisine"}}}]},
        {"id": "B7", "category": "jev", "name": "'Tomorrow' resolves to the first booking date",
         "state": {**taco, "asking": "date"},
         "turns": [{"message": "tomorrow", "expect": {"jev": {"choice": tomorrow, "next_question": "party"}}}]},
        {"id": "B8", "category": "jev", "name": "Date outside the window is not a choice",
         "state": {**taco, "asking": "date"},
         "turns": [{"message": "Christmas day please", "expect": {"jev": {"choice": "other", "next_question": "date"}}}]},
        {"id": "B9", "category": "jev", "name": "'Me and my wife' -> 2",
         "state": {**taco, "asking": "party", "date": taco_day},
         "turns": [{"message": "me and my wife", "expect": {"jev": {"choice": "2", "next_question": "time"}}}]},
        {"id": "B10", "category": "jev", "name": "Party of 5 is not rounded to 4 or 6",
         "state": {**taco, "asking": "party", "date": taco_day},
         "turns": [{"message": "five of us", "expect": {"jev": {"choice": "other", "next_question": "party"}}}]},
        {"id": "B11", "category": "jev", "name": "'Dinner at 7' -> 19:00 slot",
         "state": {**taco, "asking": "time", "date": taco_day, "party": 4},
         "turns": [{"message": "dinner at 7", "expect": {"jev": {
             "choice": "19:00", "next_question": "cuisine", "tool_output_has": "has availability"}}}]},
        {"id": "B12", "category": "jev", "name": "Off-topic stays on the question",
         "state": {**taco, "asking": "party", "date": taco_day},
         "turns": [{"message": "what's the weather today?", "expect": {"jev": {
             "choice": "other", "next_question": "party", "no_tools": True}}}]},

        # ---------------- Gemini: tool use and grounded answers ----------------
        {"id": "C1", "category": "gemini", "name": "Where can I get biryani?",
         "turns": [{"message": "Where can I get biryani?", "expect": {"gemini": {
             "tools": [{"name": "search_restaurants_by_cuisine", "args": {"cuisine_type": "Indian"}}],
             "reply_has": ["Spice Palace", "Taj Flavors"],
             "judge": "Recommends Spice Palace and Taj Flavors for biryani, and does not claim Curry House serves biryani.",
         }}}]},
        {"id": "C2", "category": "gemini", "name": "Where can I get dim sum?",
         "turns": [{"message": "Where can I get dim sum?", "expect": {"gemini": {"reply_has": ["Golden Wok"]}}}]},
        {"id": "C3", "category": "gemini", "name": "Sushi is not available",
         "turns": [{"message": "Where can I get sushi?", "expect": {"gemini": {
             "no_tools": ["check_reservation_availability"],
             "judge": "Says there is no sushi or Japanese restaurant it can offer, and does not invent one.",
         }}}]},
        {"id": "C4", "category": "gemini", "name": "Phone number lookup",
         "turns": [{"message": "What's Dragon House's phone number?", "expect": {"gemini": {
             "tools": [{"name": "get_restaurant_details", "args": {"restaurant_name": "Dragon House"}}],
             "reply_has": ["9111222333"],
         }}}]},
        {"id": "C5", "category": "gemini", "name": "Opening hours reasoning",
         "turns": [{"message": "Is Pizza Bella open at 11 AM?", "expect": {"gemini": {
             "tools": [{"name": "get_restaurant_details", "args": {"restaurant_name": "Pizza Bella"}}],
             "judge": "Says Pizza Bella is not open at 11 AM because it opens at 12 PM.",
         }}}]},
        {"id": "C6", "category": "gemini", "name": "Booking without date or party size",
         "turns": [{"message": "Book me a table at Spice Palace", "expect": {"gemini": {
             "no_tools": ["check_reservation_availability"],
             "judge": "Asks for the date and the number of people instead of guessing them.",
         }}}]},
        {"id": "C7", "category": "gemini", "name": "No invented prices",
         "turns": [{"message": "How much does a meal at Curry House cost?", "expect": {"gemini": {
             "judge": "Does not state any specific price amount; at most describes the price level (₹₹) or says it doesn't know.",
         }}}]},
        {"id": "C8", "category": "gemini", "name": "'Tomorrow' resolved to a real date",
         "turns": [{"message": "Can I get a table for 4 at Pizza Bella tomorrow?", "expect": {"gemini": {
             "tools": [{"name": "check_reservation_availability",
                        "args": {"restaurant_name": "Pizza Bella", "date": tomorrow, "party_size": 4}}],
         }}}]},
        {"id": "C9", "category": "gemini", "name": "Closed day",
         "turns": [{"message": f"Is there a table for 2 at Spice Palace on {_long(monday)}?", "expect": {"gemini": {
             "tools": [{"name": "check_reservation_availability",
                        "args": {"restaurant_name": "Spice Palace", "date": monday, "party_size": 2}}],
             "judge": "Says Spice Palace is closed on that day.",
         }}}]},
        {"id": "C10", "category": "gemini", "name": "Date outside the booking window",
         "turns": [{"message": "Can I book Ming's Kitchen for 2 on 2024-11-15?", "expect": {"gemini": {
             "judge": "Says that date cannot be booked because reservations are only open for the next few days; does not say the restaurant is fully booked.",
         }}}]},
        {"id": "C11", "category": "gemini", "name": "Specific time slot",
         "turns": [{"message": f"Is there a table for 6 at Taj Flavors at 9 PM on {_long(taj_day)}?", "expect": {"gemini": {
             "tools": [{"name": "check_reservation_availability",
                        "args": {"restaurant_name": "Taj Flavors", "date": taj_day, "party_size": 6, "time": "21:00"}}],
             "judge": "Says a table for 6 is available at 9 PM.",
         }}}]},

        # ---------------- Gemini + FDA menu rating MCP tools ----------------
        {"id": "D1", "category": "fda", "name": "Grade a restaurant's speciality",
         "turns": [{"message": "How healthy is the Margherita Pizza at Pizza Bella?", "expect": {"gemini": {
             "tools_any": ["rate_dish", "rate_restaurant_menu", "explain_dish_rating"],
             "judge": "Gives the Margherita Pizza an A-E grade and says the grade is unofficial "
                      "(the FDA does not rate restaurants or dishes).",
         }}}]},
        {"id": "D2", "category": "fda", "name": "Known nutrition is rated, not estimated (score 19, E)",
         "turns": [{"message": "Rate a burger with 850 calories, 22 g saturated fat, 1650 mg sodium, "
                               "9 g added sugar, 2 g fiber and 48 g protein.", "expect": {"gemini": {
             "tools": [{"name": "rate_dish", "args": {"calories": 850, "satFat": 22, "sodium": 1650,
                                                      "sugar": 9, "fiber": 2, "protein": 48}}],
             "no_tools": ["estimate_dish_nutrition"],
             "tool_output_has": {"rate_dish": '"grade": "E"'},
             "judge": "Says the burger gets grade E (score 19) and mentions high saturated fat or sodium.",
         }}}]},
        {"id": "D3", "category": "fda", "name": "Whole menu rated from the restaurant's dishes",
         "turns": [{"message": "Rate the whole menu of Pasta Dreams for health.", "expect": {"gemini": {
             "tools": [{"name": "get_restaurant_details", "args": {"restaurant_name": "Pasta Dreams"}},
                       {"name": "rate_restaurant_menu"}],
             "tool_output_has": {"rate_restaurant_menu": '"overall"'},
             "reply_has": ["Truffle Pasta", "Lasagna"],
             "judge": "Gives Pasta Dreams an overall grade and a grade for each of Truffle Pasta and Lasagna.",
         }}}]},
        {"id": "D4", "category": "fda", "name": "Explains why a dish got its grade",
         "turns": [{"message": "Explain why the Butter Chicken at Spice Palace gets its health grade, "
                               "and how to make it healthier.", "expect": {"gemini": {
             "tools_any": ["explain_dish_rating"],
             "judge": "Explains the Butter Chicken's grade from its nutrition and suggests one healthier change.",
         }}}]},
        {"id": "D5", "category": "fda", "name": "No grade for a restaurant that doesn't exist",
         "turns": [{"message": "How healthy is the menu at Sushi Zen?", "expect": {"gemini": {
             "no_tools": ["rate_restaurant_menu"],
             "judge": "Says it has no restaurant called Sushi Zen, and does not invent a menu or a grade for it.",
         }}}]},
        {"id": "D6", "category": "fda", "name": "Does not claim FDA approval",
         "turns": [{"message": "Is Taj Flavors FDA approved?", "expect": {"gemini": {
             "judge": "Says the FDA does not rate or approve restaurants, and does not claim Taj Flavors "
                      "is FDA approved.",
         }}}]},
        {"id": "D7", "category": "fda", "name": "Booking and health question in one message",
         "turns": [{"message": "Is there a table for 2 at Pizza Bella tomorrow, and how healthy is "
                               "its Margherita Pizza?", "expect": {"gemini": {
             "tools": [{"name": "check_reservation_availability",
                        "args": {"restaurant_name": "Pizza Bella", "date": tomorrow, "party_size": 2}}],
             "tools_any": ["rate_dish", "rate_restaurant_menu", "explain_dish_rating"],
             "judge": "Answers both parts: whether a table for 2 is free tomorrow, and the pizza's health grade.",
         }}}]},
    ]
