"""
Query Router

Routes incoming queries to the appropriate pipeline using a three-tier strategy:
1. Explicit pattern matching
2. Keyword/intent mapping
3. LLM fallback for complex queries
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PipelineType(Enum):
    """Available pipeline types."""
    TEAM_PROFILE = "team_profile"
    TEAM_COMPARISON = "team_comparison"
    TEAM_TENDENCIES = "team_tendencies"
    PLAYER_STATS = "player_stats"
    PLAYER_COMPARISON = "player_comparison"
    PLAYER_RANKINGS = "player_rankings"
    SITUATION_EPA = "situation_epa"
    DECISION_ANALYSIS = "decision_analysis"
    DRIVE_SIMULATION = "drive_simulation"
    GENERAL_QUERY = "general_query"
    UNKNOWN = "unknown"


@dataclass
class RouteResult:
    """Result of routing a query."""
    pipeline: PipelineType
    confidence: float  # 0.0 to 1.0
    extracted_params: Dict[str, Any]
    tier: int  # 1, 2, or 3
    reasoning: str


# NFL team abbreviations and aliases
TEAM_ALIASES = {
    # Standard abbreviations
    'ARI': 'ARI', 'ATL': 'ATL', 'BAL': 'BAL', 'BUF': 'BUF',
    'CAR': 'CAR', 'CHI': 'CHI', 'CIN': 'CIN', 'CLE': 'CLE',
    'DAL': 'DAL', 'DEN': 'DEN', 'DET': 'DET', 'GB': 'GB',
    'HOU': 'HOU', 'IND': 'IND', 'JAX': 'JAX', 'KC': 'KC',
    'LA': 'LA', 'LAC': 'LAC', 'LAR': 'LA', 'LV': 'LV',
    'MIA': 'MIA', 'MIN': 'MIN', 'NE': 'NE', 'NO': 'NO',
    'NYG': 'NYG', 'NYJ': 'NYJ', 'PHI': 'PHI', 'PIT': 'PIT',
    'SEA': 'SEA', 'SF': 'SF', 'TB': 'TB', 'TEN': 'TEN', 'WAS': 'WAS',
    
    # Common names/aliases
    'CARDINALS': 'ARI', 'ARIZONA': 'ARI',
    'FALCONS': 'ATL', 'ATLANTA': 'ATL',
    'RAVENS': 'BAL', 'BALTIMORE': 'BAL',
    'BILLS': 'BUF', 'BUFFALO': 'BUF',
    'PANTHERS': 'CAR', 'CAROLINA': 'CAR',
    'BEARS': 'CHI', 'CHICAGO': 'CHI',
    'BENGALS': 'CIN', 'CINCINNATI': 'CIN',
    'BROWNS': 'CLE', 'CLEVELAND': 'CLE',
    'COWBOYS': 'DAL', 'DALLAS': 'DAL',
    'BRONCOS': 'DEN', 'DENVER': 'DEN',
    'LIONS': 'DET', 'DETROIT': 'DET',
    'PACKERS': 'GB', 'GREEN BAY': 'GB', 'GREENBAY': 'GB',
    'TEXANS': 'HOU', 'HOUSTON': 'HOU',
    'COLTS': 'IND', 'INDIANAPOLIS': 'IND',
    'JAGUARS': 'JAX', 'JACKSONVILLE': 'JAX', 'JAGS': 'JAX',
    'CHIEFS': 'KC', 'KANSAS CITY': 'KC',
    'RAMS': 'LA', 'LOS ANGELES RAMS': 'LA',
    'CHARGERS': 'LAC', 'LOS ANGELES CHARGERS': 'LAC',
    'RAIDERS': 'LV', 'LAS VEGAS': 'LV', 'OAKLAND': 'LV',
    'DOLPHINS': 'MIA', 'MIAMI': 'MIA',
    'VIKINGS': 'MIN', 'MINNESOTA': 'MIN',
    'PATRIOTS': 'NE', 'NEW ENGLAND': 'NE', 'PATS': 'NE',
    'SAINTS': 'NO', 'NEW ORLEANS': 'NO',
    'GIANTS': 'NYG', 'NEW YORK GIANTS': 'NYG',
    'JETS': 'NYJ', 'NEW YORK JETS': 'NYJ',
    'EAGLES': 'PHI', 'PHILADELPHIA': 'PHI', 'PHILLY': 'PHI',
    'STEELERS': 'PIT', 'PITTSBURGH': 'PIT',
    'SEAHAWKS': 'SEA', 'SEATTLE': 'SEA',
    '49ERS': 'SF', 'NINERS': 'SF', 'SAN FRANCISCO': 'SF',
    'BUCCANEERS': 'TB', 'BUCS': 'TB', 'TAMPA BAY': 'TB', 'TAMPA': 'TB',
    'TITANS': 'TEN', 'TENNESSEE': 'TEN',
    'COMMANDERS': 'WAS', 'WASHINGTON': 'WAS', 'REDSKINS': 'WAS',
    
    # Singular forms (without 's')
    'CARDINAL': 'ARI', 'FALCON': 'ATL', 'RAVEN': 'BAL', 'BILL': 'BUF',
    'PANTHER': 'CAR', 'BEAR': 'CHI', 'BENGAL': 'CIN', 'BROWN': 'CLE',
    'COWBOY': 'DAL', 'BRONCO': 'DEN', 'LION': 'DET', 'PACKER': 'GB',
    'TEXAN': 'HOU', 'COLT': 'IND', 'JAGUAR': 'JAX', 'JAG': 'JAX',
    'CHIEF': 'KC', 'RAM': 'LA', 'CHARGER': 'LAC', 'RAIDER': 'LV',
    'DOLPHIN': 'MIA', 'VIKING': 'MIN', 'PATRIOT': 'NE', 'PAT': 'NE',
    'SAINT': 'NO', 'GIANT': 'NYG', 'JET': 'NYJ', 'EAGLE': 'PHI',
    'STEELER': 'PIT', 'SEAHAWK': 'SEA', '49ER': 'SF', 'NINER': 'SF',
    'BUCCANEER': 'TB', 'BUC': 'TB', 'TITAN': 'TEN', 'COMMANDER': 'WAS',
}

# Words that should NOT be treated as teams even if they partially match
NON_TEAM_WORDS = {
    'DO', 'THE', 'A', 'AN', 'IS', 'ARE', 'HOW', 'WHAT', 'WHEN', 'WHERE',
    'WHO', 'WHY', 'MY', 'OUR', 'WE', 'US', 'THEY', 'THEIR', 'IT', 'ON',
    'AT', 'IN', 'TO', 'FOR', 'OF', 'BY', 'WITH', 'VS', 'VERSUS', 'AND',
    'OR', 'NOT', 'SHOULD', 'WOULD', 'COULD', 'CAN', 'WILL', 'PASS',
    'RUN', 'GO', 'KICK', 'PUNT', 'OFTEN', 'MUCH', 'MANY', 'TOP', 'BEST',
    'FROM', 'ABOUT', 'TELL', 'ME', 'SHOW', 'GOOD', 'BAD', 'THIS', 'THAT'
}


class QueryRouter:
    """
    Routes queries to appropriate pipelines.
    """
    
    def __init__(self):
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for Tier 1 matching."""
        
        self.tier1_patterns = [
            # Team profile - various natural language patterns
            (
                re.compile(
                    r'(?:team\s+)?profile\s+(?:for\s+)?(?P<team>\w+)',
                    re.IGNORECASE
                ),
                PipelineType.TEAM_PROFILE,
                lambda m, q: {'team': self._normalize_team(m.group('team'))}
            ),
            (
                re.compile(
                    r'(?:tell\s+me\s+about|analyze|show)\s+(?:the\s+)?(?P<team>\w+)(?:\s+offense|\s+defense)?$',
                    re.IGNORECASE
                ),
                PipelineType.TEAM_PROFILE,
                lambda m, q: {'team': self._normalize_team(m.group('team'))}
            ),
            # "how good are the Chiefs?" / "how is KC doing?"
            (
                re.compile(
                    r'how\s+(?:good|strong|are|is)\s+(?:the\s+)?(?P<team>\w+)(?:\s+doing)?',
                    re.IGNORECASE
                ),
                PipelineType.TEAM_PROFILE,
                lambda m, q: {'team': self._normalize_team(m.group('team'))}
            ),
            # "Chiefs stats" / "KC offense"
            (
                re.compile(
                    r'^(?:the\s+)?(?P<team>\w+)\s+(?:stats|statistics|numbers|offense|defense|overview)$',
                    re.IGNORECASE
                ),
                PipelineType.TEAM_PROFILE,
                lambda m, q: {'team': self._normalize_team(m.group('team'))}
            ),
            
            # Team comparison - "match up" pattern MUST come before generic "against" pattern
            # Otherwise "Eagles match up against Dallas" matches as "up against Dallas"
            (
                re.compile(
                    r'(?:how\s+do\s+)?(?:the\s+)?(?P<team1>\w+)\s+match\s+up\s+(?:against|vs\.?)\s+(?:the\s+)?(?P<team2>\w+)',
                    re.IGNORECASE
                ),
                PipelineType.TEAM_COMPARISON,
                lambda m, q: {
                    'team1': self._normalize_team(m.group('team1')),
                    'team2': self._normalize_team(m.group('team2'))
                }
            ),
            # "compare Chiefs and Bills" / "Chiefs and Bills comparison"
            (
                re.compile(
                    r'compare\s+(?:the\s+)?(?P<team1>\w+)\s+(?:and|to|with)\s+(?:the\s+)?(?P<team2>\w+)',
                    re.IGNORECASE
                ),
                PipelineType.TEAM_COMPARISON,
                lambda m, q: {
                    'team1': self._normalize_team(m.group('team1')),
                    'team2': self._normalize_team(m.group('team2'))
                }
            ),
            # "Chiefs and Bills" / "KC Bills" - just two team names together
            (
                re.compile(
                    r'^(?:the\s+)?(?P<team1>\w+)\s+(?:and|&)\s+(?:the\s+)?(?P<team2>\w+)(?:\s+comparison)?$',
                    re.IGNORECASE
                ),
                PipelineType.TEAM_COMPARISON,
                lambda m, q: {
                    'team1': self._normalize_team(m.group('team1')),
                    'team2': self._normalize_team(m.group('team2'))
                }
            ),
            # "who's better Chiefs or Bills?" / "who would win KC vs SF?"
            (
                re.compile(
                    r"(?:who(?:'s|s)?\s+better|who\s+(?:would\s+)?win(?:s)?)\s+(?:the\s+)?(?P<team1>\w+)\s+(?:or|vs\.?|versus)\s+(?:the\s+)?(?P<team2>\w+)",
                    re.IGNORECASE
                ),
                PipelineType.TEAM_COMPARISON,
                lambda m, q: {
                    'team1': self._normalize_team(m.group('team1')),
                    'team2': self._normalize_team(m.group('team2'))
                }
            ),
            # Generic comparison - "X vs Y", "X against Y" (after match up pattern)
            (
                re.compile(
                    r'(?P<team1>\w+)\s+(?:vs\.?|versus|against|compared?\s+to)\s+(?P<team2>\w+)',
                    re.IGNORECASE
                ),
                PipelineType.TEAM_COMPARISON,
                lambda m, q: {
                    'team1': self._normalize_team(m.group('team1')),
                    'team2': self._normalize_team(m.group('team2'))
                }
            ),
            
            # Situation EPA - run vs pass (with full yardline extraction)
            (
                re.compile(
                    r'(?:should\s+(?:I|we)\s+)?(?:run\s+or\s+pass|pass\s+or\s+run).*?'
                    r'(?P<down>\d)(?:st|nd|rd|th)?\s+(?:and|&)\s+(?P<distance>\d+)',
                    re.IGNORECASE
                ),
                PipelineType.SITUATION_EPA,
                lambda m, q: {
                    'down': int(m.group('down')),
                    'distance': int(m.group('distance')),
                    'yardline': self._extract_yardline(q),
                    'defenders_in_box': self._extract_defenders_in_box(q)
                }
            ),
            (
                re.compile(
                    r'(?P<down>\d)(?:st|nd|rd|th)?\s+(?:and|&)\s+(?P<distance>\d+).*?'
                    r'(?:run\s+or\s+pass|pass\s+or\s+run|what\s+(?:play|should))',
                    re.IGNORECASE
                ),
                PipelineType.SITUATION_EPA,
                lambda m, q: {
                    'down': int(m.group('down')),
                    'distance': int(m.group('distance')),
                    'yardline': self._extract_yardline(q),
                    'defenders_in_box': self._extract_defenders_in_box(q)
                }
            ),
            # Situation EPA - "should I pass on 1st and 10"
            (
                re.compile(
                    r'(?:should\s+(?:I|we)\s+)?(?:pass|run)\s+(?:on\s+)?'
                    r'(?P<down>\d)(?:st|nd|rd|th)?\s+(?:and|&)\s+(?P<distance>\d+)',
                    re.IGNORECASE
                ),
                PipelineType.SITUATION_EPA,
                lambda m, q: {
                    'down': int(m.group('down')),
                    'distance': int(m.group('distance')),
                    'yardline': self._extract_yardline(q),
                    'defenders_in_box': self._extract_defenders_in_box(q)
                }
            ),
            # Situation EPA - "3rd and 7 from the 25" format (standalone)
            (
                re.compile(
                    r'^(?P<down>\d)(?:st|nd|rd|th)?\s+(?:and|&)\s+(?P<distance>\d+)'
                    r'\s+(?:from|at|on)\s+(?:the\s+)?',
                    re.IGNORECASE
                ),
                PipelineType.SITUATION_EPA,
                lambda m, q: {
                    'down': int(m.group('down')),
                    'distance': int(m.group('distance')),
                    'yardline': self._extract_yardline(q),
                    'defenders_in_box': self._extract_defenders_in_box(q)
                }
            ),
            # Goal line situations - "2nd and goal from the 3"
            (
                re.compile(
                    r'(?P<down>\d)(?:st|nd|rd|th)?\s+(?:and|&)\s+goal',
                    re.IGNORECASE
                ),
                PipelineType.SITUATION_EPA,
                lambda m, q: {
                    'down': int(m.group('down')),
                    'distance': self._extract_yardline(q) or 5,  # distance = yardline for goal
                    'yardline': self._extract_yardline(q) or 5
                }
            ),
            
            # Natural language: "run or pass at the X yard line on Yth down"
            # Handles queries where yardline and down are specified separately (no explicit distance)
            (
                re.compile(
                    r'(?:should\s+(?:I|we)\s+)?(?:run\s+or\s+pass|pass\s+or\s+run).*?'
                    r'(?:at|on|from)\s+(?:the\s+)?(?P<yardline>\d+)\s*(?:yard\s*line)?.*?'
                    r'(?:on\s+)?(?P<down>\d)(?:st|nd|rd|th)?\s+down',
                    re.IGNORECASE
                ),
                PipelineType.SITUATION_EPA,
                lambda m, q: {
                    'down': int(m.group('down')),
                    'distance': min(int(m.group('yardline')), 10),  # distance = yardline (capped at 10) for goal line
                    'yardline': int(m.group('yardline')),
                    'defenders_in_box': self._extract_defenders_in_box(q)
                }
            ),
            # Reversed order: "on Yth down at the X yard line"
            (
                re.compile(
                    r'(?:should\s+(?:I|we)\s+)?(?:run\s+or\s+pass|pass\s+or\s+run).*?'
                    r'(?:on\s+)?(?P<down>\d)(?:st|nd|rd|th)?\s+down.*?'
                    r'(?:at|on|from)\s+(?:the\s+)?(?P<yardline>\d+)\s*(?:yard\s*line)?',
                    re.IGNORECASE
                ),
                PipelineType.SITUATION_EPA,
                lambda m, q: {
                    'down': int(m.group('down')),
                    'distance': min(int(m.group('yardline')), 10),
                    'yardline': int(m.group('yardline')),
                    'defenders_in_box': self._extract_defenders_in_box(q)
                }
            ),
            # Simple: "Xth down at the Y" (no explicit run/pass but has down and yardline)
            (
                re.compile(
                    r'(?P<down>\d)(?:st|nd|rd|th)?\s+down\s+(?:at|on|from)\s+(?:the\s+)?(?P<yardline>\d+)',
                    re.IGNORECASE
                ),
                PipelineType.SITUATION_EPA,
                lambda m, q: {
                    'down': int(m.group('down')),
                    'distance': min(int(m.group('yardline')), 10),
                    'yardline': int(m.group('yardline')),
                    'defenders_in_box': self._extract_defenders_in_box(q)
                }
            ),
            
            # 4th down decision - improved yardline extraction
            (
                re.compile(
                    r'(?:should\s+(?:I|we)\s+)?go\s+for\s+it.*?'
                    r'4th\s+(?:and|&)\s+(?P<distance>\d+)',
                    re.IGNORECASE
                ),
                PipelineType.DECISION_ANALYSIS,
                lambda m, q: {
                    'down': 4,
                    'distance': int(m.group('distance')),
                    'yardline': self._extract_yardline(q)
                }
            ),
            (
                re.compile(
                    r'4th\s+(?:and|&)\s+(?P<distance>\d+)',
                    re.IGNORECASE
                ),
                PipelineType.DECISION_ANALYSIS,
                lambda m, q: {
                    'down': 4,
                    'distance': int(m.group('distance')),
                    'yardline': self._extract_yardline(q)
                }
            ),
            # 4th and goal
            (
                re.compile(
                    r'4th\s+(?:and|&)\s+goal',
                    re.IGNORECASE
                ),
                PipelineType.DECISION_ANALYSIS,
                lambda m, q: {
                    'down': 4,
                    'distance': self._extract_yardline(q) or 1,
                    'yardline': self._extract_yardline(q) or 1
                }
            ),
            
            # Player rankings - improved position matching
            (
                re.compile(
                    r'(?:top|best|leading)\s+(?P<count>\d+)?\s*'
                    r'(?P<position>QB|RB|WR|TE|qb|rb|wr|te|quarterbacks?|running\s*backs?|wide\s*receivers?|receivers?|tight\s*ends?)s?'
                    r'(?:\s+by\s+(?P<metric>EPA|epa|yards|touchdowns?))?',
                    re.IGNORECASE
                ),
                PipelineType.PLAYER_RANKINGS,
                lambda m, q: {
                    'count': int(m.group('count')) if m.group('count') else 10,
                    'position': self._normalize_position(m.group('position')),
                    'metric': m.group('metric').lower() if m.group('metric') else 'epa'
                }
            ),
            
            # Team tendencies - use word boundary to prevent matching mid-word (e.g., "ive" from "offensive")
            (
                re.compile(
                    r'\b(?P<team>ravens?|chiefs?|eagles?|cowboys?|bills?|dolphins?|patriots?|jets?|'
                    r'steelers?|browns?|bengals?|49ers?|niners?|seahawks?|rams?|cardinals?|'
                    r'chargers?|broncos?|raiders?|texans?|colts?|jaguars?|titans?|'
                    r'saints?|falcons?|panthers?|buccaneers?|bucs?|packers?|bears?|lions?|vikings?|'
                    r'giants?|commanders?|[A-Z]{2,3})\s+'
                    r'(?:tendenc(?:y|ies)|style|play\s*calling)',
                    re.IGNORECASE
                ),
                PipelineType.TEAM_TENDENCIES,
                lambda m, q: {'team': self._normalize_team(m.group('team'))}
            ),
            (
                re.compile(
                    r'(?:what\s+is\s+(?:the\s+)?)?'
                    r'\b(?P<team>KC|SF|BAL|BUF|MIA|NE|NYJ|NYG|PHI|DAL|WAS|CHI|GB|MIN|DET|'
                    r'TB|NO|ATL|CAR|LAR?|SEA|ARI|LA|LAC|DEN|LV|HOU|IND|JAX|TEN|CIN|CLE|PIT)\b\s+'
                    r'(?:offensive\s+)?style',
                    re.IGNORECASE
                ),
                PipelineType.TEAM_TENDENCIES,
                lambda m, q: {'team': self._normalize_team(m.group('team'))}
            ),
        ]
        
        # Tier 2: Keyword-based routing
        self.tier2_keywords = {
            PipelineType.TEAM_PROFILE: [
                'profile', 'about', 'tell me', 'analyze', 'how good',
                'offense', 'defense', 'strengths', 'weaknesses'
            ],
            PipelineType.TEAM_COMPARISON: [
                'vs', 'versus', 'compare', 'matchup', 'match up', 'against', 'better team'
            ],
            PipelineType.TEAM_TENDENCIES: [
                'tendency', 'tendencies', 'how often', 'likely', 'usually',
                'prefer', 'style', 'play calling'
            ],
            PipelineType.SITUATION_EPA: [
                'run or pass', 'pass or run', 'what play', 'best play',
                'should call', 'recommend', 'optimal', 'expected points'
            ],
            PipelineType.DECISION_ANALYSIS: [
                'go for it', 'kick', 'punt', 'field goal', '4th down',
                'fourth down', 'decision'
            ],
            PipelineType.PLAYER_RANKINGS: [
                'top', 'best', 'leading', 'ranking', 'ranked', 'leaders'
            ],
            PipelineType.PLAYER_COMPARISON: [
                'player compare', 'who is better', 'vs player'
            ],
            PipelineType.DRIVE_SIMULATION: [
                'simulate', 'simulation', 'expected points from', 'drive'
            ],
        }
    
    def _normalize_team(self, team_str: str) -> Optional[str]:
        """Convert team name/alias to standard abbreviation."""
        if not team_str:
            return None
        normalized = team_str.upper().strip()
        
        # Skip non-team words
        if normalized in NON_TEAM_WORDS:
            return None
        
        # Direct lookup in aliases (includes all valid abbreviations)
        if normalized in TEAM_ALIASES:
            return TEAM_ALIASES[normalized]
            
        # Try without trailing 's' for plural team names (e.g., "Eagles" -> "EAGLE")
        if normalized.endswith('S') and len(normalized) > 3:
            singular = normalized[:-1]
            if singular in TEAM_ALIASES:
                return TEAM_ALIASES[singular]
        
        # Don't accept arbitrary 2-3 letter strings - must be in TEAM_ALIASES
        # This prevents words like "UP", "AT", "IN" from being treated as teams
        return None
    
    def _normalize_position(self, pos_str: str) -> str:
        """Normalize position string."""
        pos_upper = pos_str.upper().strip()
        
        # Handle multi-word positions
        pos_upper = re.sub(r'\s+', ' ', pos_upper)  # Normalize spaces
        
        pos_map = {
            'QUARTERBACK': 'QB', 'QUARTERBACKS': 'QB', 'QB': 'QB',
            'RUNNING BACK': 'RB', 'RUNNING BACKS': 'RB', 'RUNNINGBACK': 'RB',
            'RUNNINGBACKS': 'RB', 'RB': 'RB',
            'WIDE RECEIVER': 'WR', 'WIDE RECEIVERS': 'WR', 'WIDERECEIVER': 'WR',
            'WIDERECEIVERS': 'WR', 'RECEIVER': 'WR', 'RECEIVERS': 'WR', 'WR': 'WR',
            'TIGHT END': 'TE', 'TIGHT ENDS': 'TE', 'TIGHTEND': 'TE',
            'TIGHTENDS': 'TE', 'TE': 'TE',
        }
        return pos_map.get(pos_upper, 'RB')  # Default to RB if unknown
    
    def _extract_teams(self, query: str, context: Optional[Dict] = None) -> List[str]:
        """Extract team mentions from query."""
        teams = []
        query_upper = query.upper()
        
        # First check multi-word team names (higher priority)
        multi_word_teams = [
            ('GREEN BAY', 'GB'), ('KANSAS CITY', 'KC'), ('NEW ENGLAND', 'NE'),
            ('NEW ORLEANS', 'NO'), ('NEW YORK GIANTS', 'NYG'), ('NEW YORK JETS', 'NYJ'),
            ('LOS ANGELES RAMS', 'LA'), ('LOS ANGELES CHARGERS', 'LAC'),
            ('LAS VEGAS', 'LV'), ('TAMPA BAY', 'TB'), ('SAN FRANCISCO', 'SF'),
        ]
        
        for alias, abbrev in multi_word_teams:
            if alias in query_upper:
                if abbrev not in teams:
                    teams.append(abbrev)
        
        # Then check single words
        words = re.findall(r'\b\w+\b', query_upper)
        
        for word in words:
            if word in NON_TEAM_WORDS:
                continue
            if word in TEAM_ALIASES:
                abbrev = TEAM_ALIASES[word]
                if abbrev not in teams:
                    teams.append(abbrev)
        
        # Handle "we", "our", "my team" with context
        if context and context.get('favorite_team'):
            we_patterns = [r'\bwe\b', r'\bour\b', r'\bmy team\b', r'\bus\b']
            for pattern in we_patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    fav_team = context['favorite_team']
                    if fav_team not in teams:
                        teams.insert(0, fav_team)  # Put favorite team first
                    break
        
        return teams
    
    def _extract_down_distance(self, query: str) -> Tuple[Optional[int], Optional[int]]:
        """Extract down and distance from query."""
        # Handle "and goal" as distance
        goal_match = re.search(r'(\d)(?:st|nd|rd|th)?\s*(?:and|&)\s*goal', query, re.IGNORECASE)
        if goal_match:
            return int(goal_match.group(1)), None  # Distance will come from yardline
        
        match = re.search(
            r'(\d)(?:st|nd|rd|th)?\s*(?:and|&)\s*(\d+)',
            query,
            re.IGNORECASE
        )
        if match:
            return int(match.group(1)), int(match.group(2))
        return None, None
    
    def _extract_yardline(self, query: str) -> Optional[int]:
        """
        Extract yardline from query with comprehensive pattern matching.
        
        Handles:
        - "at the 35", "at my 36", "on the 40"
        - "from the 25", "from my 28"
        - "at midfield" -> 50
        - "at my own 28" -> 72 (100 - 28)
        - "on their 15", "at the opponent's 40" -> as-is
        - "40 yard line"
        """
        query_lower = query.lower()
        
        # Handle "midfield"
        if 'midfield' in query_lower:
            return 50
        
        # Handle "my own X" or "our own X" (own territory = far from opponent endzone)
        own_match = re.search(
            r'(?:my|our)\s+own\s+(\d+)',
            query_lower
        )
        if own_match:
            return 100 - int(own_match.group(1))
        
        # Handle "their X" or "opponent's X" (opponent territory)
        opponent_match = re.search(
            r'(?:their|opponent\'?s?)\s+(\d+)',
            query_lower
        )
        if opponent_match:
            return int(opponent_match.group(1))
        
        # Handle yardlines with explicit context
        # IMPORTANT: Require "the" or "my" after preposition to avoid matching ordinals like "on 3rd"
        patterns = [
            # "at the 35", "at my 36", "on the 40", "from the 25" - require "the" or "my"
            r'(?:at|on|from)\s+(?:the|my)\s+(\d{1,2})(?:\s*(?:yard(?:\s*line)?)?)?(?:\s|$|,)',
            # "35 yard line", "40 yard line" - explicit yard line mention
            r'(\d{1,2})\s+yard\s*line',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                yardline = int(match.group(1))
                # Sanity check: yardline should be 1-99
                if 1 <= yardline <= 99:
                    return yardline
        
        return None
    
    def _extract_defenders_in_box(self, query: str) -> Optional[int]:
        """
        Extract number of defenders in the box from query.
        
        Examples:
        - "8 in the box" -> 8
        - "stacked box" -> 8
        - "light box" / "6 man box" -> 6
        - "7 defenders" -> 7
        """
        query_lower = query.lower()
        
        # Explicit number patterns
        patterns = [
            r'(\d)\s*(?:men?|defenders?)?\s*(?:in\s*)?(?:the\s*)?box',
            r'(\d)\s*man\s+box',
            r'box\s+(?:with\s+)?(\d)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                count = int(match.group(1))
                if 5 <= count <= 9:  # Valid range
                    return count
        
        # Named patterns
        if 'stacked box' in query_lower or 'loaded box' in query_lower:
            return 8
        if 'light box' in query_lower:
            return 6
        
        return None
    
    def _handle_followup(self, query: str, context: Dict) -> Optional[RouteResult]:
        """
        Handle follow-up questions that reference previous conversation context.
        
        Patterns handled:
        - "what about the Cowboys?" -> uses last_pipeline with new team
        - "and on 3rd down?" -> adds down filter to last query
        - "how about rushing?" -> modifies last query
        - "compare them to the Bills" -> comparison with last_team
        - "what about at the 20?" -> changes yardline
        
        Args:
            query: User's query
            context: Context including conversation history
            
        Returns:
            RouteResult if this is a follow-up, None otherwise
        """
        query_lower = query.lower().strip()
        
        # Get conversation history from context
        history = context.get('history', {})
        last_team = history.get('last_team')
        last_team2 = history.get('last_team2')
        last_pipeline = history.get('last_pipeline')
        last_down = history.get('last_down')
        last_distance = history.get('last_distance')
        last_yardline = history.get('last_yardline')
        last_position = history.get('last_position')
        
        # No history to follow up on
        if not last_pipeline:
            return None
        
        # Follow-up patterns - more flexible matching
        followup_patterns = [
            # "what about X?" / "how about X?" / "and X?"
            (r'^(?:what|how)\s+about\s+(?:the\s+)?(.+?)\??$', 'topic_change'),
            (r'^and\s+(?:the\s+)?(.+?)\??$', 'additive'),
            (r'^(?:now\s+)?(?:for|with)\s+(?:the\s+)?(.+?)\??$', 'topic_change'),
            # "compare them to X" / "vs X" / "contrast with X" - more flexible
            (r'^(?:compare\s+(?:them|that|it)?\s*(?:to|with|against)|contrast\s*(?:them|that|it)?\s*(?:with|to|against)?|vs\.?)\s*(?:the\s+)?(.+?)$', 'compare_to'),
            (r'^(?:how\s+do\s+they\s+compare\s+(?:to|with)|stack\s+(?:them\s+)?up\s+against)\s+(?:the\s+)?(.+?)$', 'compare_to'),
            # Simpler contrast patterns
            (r'^contrast\s+(?:with\s+)?(?:the\s+)?(.+?)$', 'compare_to'),
            (r'^(?:compare|match)\s+(?:with\s+)?(?:the\s+)?(.+?)$', 'compare_to'),
            # "on Xth down" / "at the X"
            (r'^(?:and\s+)?on\s+(\d)(?:st|nd|rd|th)\s+(?:down)?', 'down_filter'),
            (r'^(?:and\s+)?(?:at|from)\s+(?:the\s+)?(\d+)', 'yardline_change'),
            # "what about rushing/passing?"
            (r'^(?:what|how)\s+about\s+(rush(?:ing)?|pass(?:ing)?|run(?:ning)?)', 'play_type'),
        ]
        
        for pattern, followup_type in followup_patterns:
            match = re.match(pattern, query_lower)
            if match:
                captured = match.group(1).strip() if match.groups() else None
                
                # Handle different follow-up types
                if followup_type == 'topic_change':
                    # Try to extract a team from the captured text
                    new_team = self._normalize_team(captured) if captured else None
                    
                    if new_team:
                        # Same pipeline, different team
                        params = {'team': new_team, 'season': context.get('season', 2025)}
                        
                        # For comparison, keep one team and swap other
                        if last_pipeline == 'team_comparison' and last_team:
                            params = {
                                'team1': last_team,
                                'team2': new_team,
                                'season': context.get('season', 2025)
                            }
                            return RouteResult(
                                pipeline=PipelineType.TEAM_COMPARISON,
                                confidence=0.85,
                                extracted_params=params,
                                tier=1,
                                reasoning=f"Follow-up comparison: {last_team} vs {new_team}"
                            )
                        
                        # Determine pipeline from last or default to profile
                        pipeline = PipelineType.TEAM_PROFILE
                        if last_pipeline == 'team_tendencies':
                            pipeline = PipelineType.TEAM_TENDENCIES
                        elif last_pipeline == 'team_profile':
                            pipeline = PipelineType.TEAM_PROFILE
                        
                        return RouteResult(
                            pipeline=pipeline,
                            confidence=0.85,
                            extracted_params=params,
                            tier=1,
                            reasoning=f"Follow-up: same analysis for {new_team}"
                        )
                    
                    # Check if it's a position change for player rankings
                    new_position = self._normalize_position(captured) if captured else None
                    if new_position and last_pipeline == 'player_rankings':
                        return RouteResult(
                            pipeline=PipelineType.PLAYER_RANKINGS,
                            confidence=0.85,
                            extracted_params={
                                'position': new_position,
                                'season': context.get('season', 2025)
                            },
                            tier=1,
                            reasoning=f"Follow-up: player rankings for {new_position}"
                        )
                
                elif followup_type == 'compare_to' and last_team:
                    new_team = self._normalize_team(captured) if captured else None
                    if new_team:
                        return RouteResult(
                            pipeline=PipelineType.TEAM_COMPARISON,
                            confidence=0.85,
                            extracted_params={
                                'team1': last_team,
                                'team2': new_team,
                                'season': context.get('season', 2025)
                            },
                            tier=1,
                            reasoning=f"Follow-up comparison: {last_team} vs {new_team}"
                        )
                
                elif followup_type == 'down_filter' and last_team:
                    new_down = int(captured) if captured and captured.isdigit() else None
                    if new_down:
                        return RouteResult(
                            pipeline=PipelineType.TEAM_TENDENCIES,
                            confidence=0.85,
                            extracted_params={
                                'team': last_team,
                                'down': new_down,
                                'season': context.get('season', 2025)
                            },
                            tier=1,
                            reasoning=f"Follow-up: {last_team} tendencies on {new_down} down"
                        )
                
                elif followup_type == 'yardline_change':
                    new_yardline = int(captured) if captured and captured.isdigit() else None
                    if new_yardline and last_down and last_distance:
                        return RouteResult(
                            pipeline=PipelineType.SITUATION_EPA,
                            confidence=0.85,
                            extracted_params={
                                'down': last_down,
                                'distance': last_distance,
                                'yardline': new_yardline,
                                'season': context.get('season', 2025)
                            },
                            tier=1,
                            reasoning=f"Follow-up: same situation at {new_yardline} yard line"
                        )
        
        return None
    
    def route(self, query: str, context: Optional[Dict] = None) -> RouteResult:
        """
        Route a query to the appropriate pipeline.
        
        Args:
            query: User's query string
            context: Optional context (favorite team, season, conversation history, etc.)
            
        Returns:
            RouteResult with pipeline and extracted parameters
        """
        context = context or {}
        
        # Check for follow-up questions first
        followup_result = self._handle_followup(query, context)
        if followup_result:
            return followup_result
        
        # Tier 1: Explicit pattern matching
        for pattern, pipeline_type, param_extractor in self.tier1_patterns:
            match = pattern.search(query)
            if match:
                try:
                    params = param_extractor(match, query)
                    
                    # Filter out None values from params
                    params = {k: v for k, v in params.items() if v is not None}
                    
                    # Add context
                    if 'season' not in params:
                        params['season'] = context.get('season', 2025)
                    
                    # Validate team params
                    if params.get('team') is None and pipeline_type in [
                        PipelineType.TEAM_PROFILE, 
                        PipelineType.TEAM_TENDENCIES
                    ]:
                        # Try to extract team from full query
                        teams = self._extract_teams(query, context)
                        if teams:
                            params['team'] = teams[0]
                        elif context.get('favorite_team'):
                            params['team'] = context['favorite_team']
                        else:
                            continue  # Skip this pattern, try next
                    
                    # Validate comparison params
                    if pipeline_type == PipelineType.TEAM_COMPARISON:
                        if not params.get('team1') or not params.get('team2'):
                            teams = self._extract_teams(query, context)
                            if len(teams) >= 2:
                                params['team1'] = teams[0]
                                params['team2'] = teams[1]
                            elif len(teams) == 1 and context.get('favorite_team'):
                                params['team1'] = context['favorite_team']
                                params['team2'] = teams[0]
                            else:
                                continue  # Skip this pattern
                    
                    return RouteResult(
                        pipeline=pipeline_type,
                        confidence=0.95,
                        extracted_params=params,
                        tier=1,
                        reasoning=f"Matched pattern for {pipeline_type.value}"
                    )
                except Exception as e:
                    logger.debug(f"Pattern matched but param extraction failed: {e}")
                    continue
        
        # Tier 2: Keyword-based intent
        query_lower = query.lower()
        best_match = None
        best_score = 0
        
        for pipeline_type, keywords in self.tier2_keywords.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > best_score:
                best_score = score
                best_match = pipeline_type
        
        if best_match and best_score >= 1:
            # Extract what we can
            params = {'season': context.get('season', 2025)}
            
            teams = self._extract_teams(query, context)
            if teams:
                params['team'] = teams[0]
                if len(teams) > 1:
                    params['team1'] = teams[0]
                    params['team2'] = teams[1]
            
            down, distance = self._extract_down_distance(query)
            if down:
                params['down'] = down
            if distance:
                params['distance'] = distance
            
            yardline = self._extract_yardline(query)
            if yardline:
                params['yardline'] = yardline
            
            # For goal line, distance = yardline
            if down and not distance and yardline:
                params['distance'] = yardline
            
            # Use context team if no team found
            if 'team' not in params and context.get('favorite_team'):
                params['team'] = context['favorite_team']
            
            return RouteResult(
                pipeline=best_match,
                confidence=min(0.6 + best_score * 0.1, 0.85),
                extracted_params=params,
                tier=2,
                reasoning=f"Keyword match for {best_match.value} (score: {best_score})"
            )
        
        # Tier 3: General query (needs LLM or more context)
        params = {'season': context.get('season', 2025)}
        teams = self._extract_teams(query, context)
        if teams:
            params['teams'] = teams
        
        return RouteResult(
            pipeline=PipelineType.GENERAL_QUERY,
            confidence=0.3,
            extracted_params=params,
            tier=3,
            reasoning="No clear pattern match, routing to general handler"
        )
    
    def route_with_suggestions(self, query: str, context: Optional[Dict] = None) -> Dict:
        """
        Route query and provide suggestions if confidence is low.
        """
        result = self.route(query, context)
        
        response = {
            'route': result,
            'suggestions': []
        }
        
        if result.confidence < 0.7:
            teams = self._extract_teams(query, context)
            
            if not teams and result.pipeline in [
                PipelineType.TEAM_PROFILE, 
                PipelineType.TEAM_TENDENCIES
            ]:
                response['suggestions'].append(
                    "Which team would you like me to analyze?"
                )
            
            down, distance = self._extract_down_distance(query)
            if result.pipeline == PipelineType.SITUATION_EPA and not down:
                response['suggestions'].append(
                    "What's the down and distance? (e.g., '3rd and 5')"
                )
            
            if result.pipeline == PipelineType.DECISION_ANALYSIS:
                if 'yardline' not in result.extracted_params:
                    response['suggestions'].append(
                        "What yard line are you at?"
                    )
        
        return response
