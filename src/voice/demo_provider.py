"""Demo and synthetic emergency call provider with realistic caller scenarios."""

from src.voice.interface import VoiceProvider

DEMO_EMERGENCY_CALLS = [
    # Traffic Gridlock Bypass Demo (P3 Critical)
    {
        "id": "demo_p3_traffic_bypass",
        "text": "CRITICAL EMERGENCY! 60-year-old male sudden cardiac arrest at 4th and Market Downtown! Not breathing, CPR in progress!",
        "expected_prio": 3,
        "type": "CARDIAC",
        "location": (75.0, 60.0),
        "is_traffic_demo": True,
    },
    # Priority 3 Critical Calls
    {
        "id": "demo_p3_bike",
        "text": "There has been a serious bike accident. The person is unconscious and bleeding badly.",
        "expected_prio": 3,
        "type": "TRAFFIC_ACCIDENT",
        "location": (72.0, 31.0),
    },
    {
        "id": "demo_p3_cardiac",
        "text": "My father has collapsed on the kitchen floor! He is not responding and clutching his chest!",
        "expected_prio": 3,
        "type": "CARDIAC",
        "location": (28.5, 64.0),
    },
    {
        "id": "demo_p3_choking",
        "text": "My 4-year-old is choking on food! His lips are turning blue, he's not breathing!",
        "expected_prio": 3,
        "type": "BREATHING",
        "location": (88.0, 15.0),
    },
    {
        "id": "demo_p3_fire",
        "text": "Heavy black smoke and flames coming from the apartment building! People are trapped on the second floor!",
        "expected_prio": 3,
        "type": "FIRE",
        "location": (45.0, 85.0),
    },
    {
        "id": "demo_p3_head_trauma",
        "text": "Construction worker fell off scaffolding, about 20 feet. Unconscious with severe bleeding from head.",
        "expected_prio": 3,
        "type": "INJURY",
        "location": (12.0, 38.0),
    },

    # Priority 2 Urgent Calls
    {
        "id": "demo_p2_car_crash",
        "text": "Two cars collided at the intersection. Both drivers are awake, but one has a broken leg and cannot get out.",
        "expected_prio": 2,
        "type": "TRAFFIC_ACCIDENT",
        "location": (62.0, 48.0),
    },
    {
        "id": "demo_p2_asthma",
        "text": "My teenage daughter is having a severe asthma attack. Her inhaler is empty and she is wheezing heavily.",
        "expected_prio": 2,
        "type": "BREATHING",
        "location": (33.0, 22.0),
    },
    {
        "id": "demo_p2_deep_cut",
        "text": "Accident in restaurant kitchen, chef has deep laceration on forearm with continuous heavy bleeding.",
        "expected_prio": 2,
        "type": "INJURY",
        "location": (79.0, 75.0),
    },
    {
        "id": "demo_p2_kitchen_fire",
        "text": "Stove caught fire, flames spread to the cabinets. We evacuated but someone has second degree burns.",
        "expected_prio": 2,
        "type": "FIRE",
        "location": (18.0, 88.0),
    },
    {
        "id": "demo_p2_diabetic",
        "text": "My grandfather is diabetic, he is extremely dizzy, confused, slurring speech and cannot stand up.",
        "expected_prio": 2,
        "type": "MEDICAL",
        "location": (54.0, 12.0),
    },

    # Priority 1 Routine Calls
    {
        "id": "demo_p1_ankle",
        "text": "I slipped on the ice while walking my dog and twisted my ankle badly. I can't put any weight on it.",
        "expected_prio": 1,
        "type": "INJURY",
        "location": (22.0, 15.0),
    },
    {
        "id": "demo_p1_elderly_fall",
        "text": "Elderly neighbor slipped out of bed. She is completely alert and oriented, no bleeding, but needs assistance getting up.",
        "expected_prio": 1,
        "type": "MEDICAL",
        "location": (70.0, 80.0),
    },
    {
        "id": "demo_p1_minor_fever",
        "text": "Adult with persistent high fever and stomach flu for 2 days. Dehydrated and needs evaluation.",
        "expected_prio": 1,
        "type": "MEDICAL",
        "location": (35.0, 45.0),
    },
    {
        "id": "demo_p1_minor_cut",
        "text": "Cut hand on broken glass while cleaning. Bleeding has stopped with pressure, but may need stitches.",
        "expected_prio": 1,
        "type": "INJURY",
        "location": (91.0, 60.0),
    },
]


class DemoAudioProvider(VoiceProvider):
    """Provides synthetic emergency caller transcripts for demo and simulation runs."""

    def __init__(self) -> None:
        self.call_bank = DEMO_EMERGENCY_CALLS

    def process_call(self, payload: dict) -> str:
        """Extract text from demo payload or select by ID/index."""
        if "text" in payload and payload["text"]:
            return payload["text"]

        call_id = payload.get("call_id")
        if call_id:
            for c in self.call_bank:
                if c["id"] == call_id:
                    return c["text"]

        # Default fallback call
        return self.call_bank[0]["text"]

    def get_preset(self, index: int) -> dict:
        return self.call_bank[index % len(self.call_bank)]
