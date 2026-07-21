#!/usr/bin/env python3
"""
Build a comprehensive pop culture / sports / TV / movies / music / science fact database.
Outputs JSONL for loading into FactDB.

Usage:
    python data/facts/build_popculture_facts.py
"""

import json
import os

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pop_culture_facts_v1.jsonl")

def make_fact(fid, category, question, answer):
    return {"id": fid, "category": category, "question": question, "answer": answer, "source": "pop-culture-v1"}

def build_facts():
    facts = []
    idx = 0

    def add(cat, q, a):
        nonlocal idx
        idx += 1
        facts.append(make_fact(f"pop-{idx:04d}", cat, q, a))

    # ==========================================================================
    # 1. US NAVY RANKS — complete hierarchy
    # ==========================================================================
    navy_cat = "us_navy"
    add(navy_cat, "What is the lowest enlisted rank in the US Navy?", "Seaman Recruit (E-1)")
    add(navy_cat, "What is the rank of Seaman Apprentice in the US Navy?", "E-2")
    add(navy_cat, "What is the rank of Seaman in the US Navy?", "E-3")
    add(navy_cat, "What is the rank of Petty Officer Third Class in the US Navy?", "E-4")
    add(navy_cat, "What is the rank of Petty Officer Second Class in the US Navy?", "E-5")
    add(navy_cat, "What is the rank of Petty Officer First Class in the US Navy?", "E-6")
    add(navy_cat, "What is the rank of Chief Petty Officer in the US Navy?", "E-7")
    add(navy_cat, "What is the rank of Senior Chief Petty Officer in the US Navy?", "E-8")
    add(navy_cat, "What is the rank of Master Chief Petty Officer in the US Navy?", "E-9")
    add(navy_cat, "What is the rank of Master Chief Petty Officer of the Navy in the US Navy?", "E-9 (special)")

    add(navy_cat, "What is the lowest officer rank in the US Navy?", "Ensign (O-1)")
    add(navy_cat, "What is the rank of Lieutenant Junior Grade in the US Navy?", "O-2")
    add(navy_cat, "What is the rank of Lieutenant in the US Navy?", "O-3")
    add(navy_cat, "What is the rank of Lieutenant Commander in the US Navy?", "O-4")
    add(navy_cat, "What is the rank of Commander in the US Navy?", "O-5")
    add(navy_cat, "What is the rank of Captain in the US Navy?", "O-6")
    add(navy_cat, "What is the rank of Rear Admiral Lower Half in the US Navy?", "O-7")
    add(navy_cat, "What is the rank of Rear Admiral Upper Half in the US Navy?", "O-8")
    add(navy_cat, "What is the rank of Vice Admiral in the US Navy?", "O-9")
    add(navy_cat, "What is the rank of Admiral in the US Navy?", "O-10")
    add(navy_cat, "What is the rank of Fleet Admiral in the US Navy?", "O-11 (wartime only)")

    add(navy_cat, "What are the ranks in the US Navy from lowest to highest?", "Seaman Recruit (E-1), Seaman Apprentice (E-2), Seaman (E-3), Petty Officer Third Class (E-4), Petty Officer Second Class (E-5), Petty Officer First Class (E-6), Chief Petty Officer (E-7), Senior Chief Petty Officer (E-8), Master Chief Petty Officer (E-9), Ensign (O-1), Lieutenant Junior Grade (O-2), Lieutenant (O-3), Lieutenant Commander (O-4), Commander (O-5), Captain (O-6), Rear Admiral Lower Half (O-7), Rear Admiral Upper Half (O-8), Vice Admiral (O-9), Admiral (O-10), Fleet Admiral (O-11)")
    add(navy_cat, "What is the US Navy enlisted rank for E-1?", "Seaman Recruit")
    add(navy_cat, "What is the US Navy enlisted rank for E-2?", "Seaman Apprentice")
    add(navy_cat, "What is the US Navy enlisted rank for E-3?", "Seaman")
    add(navy_cat, "What is the US Navy enlisted rank for E-4?", "Petty Officer Third Class")
    add(navy_cat, "What is the US Navy enlisted rank for E-5?", "Petty Officer Second Class")
    add(navy_cat, "What is the US Navy enlisted rank for E-6?", "Petty Officer First Class")
    add(navy_cat, "What is the US Navy enlisted rank for E-7?", "Chief Petty Officer")
    add(navy_cat, "What is the US Navy enlisted rank for E-8?", "Senior Chief Petty Officer")
    add(navy_cat, "What is the US Navy enlisted rank for E-9?", "Master Chief Petty Officer")
    add(navy_cat, "What is the US Navy officer rank for O-1?", "Ensign")
    add(navy_cat, "What is the US Navy officer rank for O-2?", "Lieutenant Junior Grade")
    add(navy_cat, "What is the US Navy officer rank for O-3?", "Lieutenant")
    add(navy_cat, "What is the US Navy officer rank for O-4?", "Lieutenant Commander")
    add(navy_cat, "What is the US Navy officer rank for O-5?", "Commander")
    add(navy_cat, "What is the US Navy officer rank for O-6?", "Captain")
    add(navy_cat, "What is the US Navy officer rank for O-7?", "Rear Admiral Lower Half")
    add(navy_cat, "What is the US Navy officer rank for O-8?", "Rear Admiral Upper Half")
    add(navy_cat, "What is the US Navy officer rank for O-9?", "Vice Admiral")
    add(navy_cat, "What is the US Navy officer rank for O-10?", "Admiral")
    add(navy_cat, "What is the US Navy officer rank for O-11?", "Fleet Admiral")

    # ==========================================================================
    # 2. HARRY POTTER BOOK SERIES
    # ==========================================================================
    hp_cat = "harry_potter"
    add(hp_cat, "What is the first Harry Potter book?", "Harry Potter and the Philosopher's Stone")
    add(hp_cat, "What is the second Harry Potter book?", "Harry Potter and the Chamber of Secrets")
    add(hp_cat, "What is the third Harry Potter book?", "Harry Potter and the Prisoner of Azkaban")
    add(hp_cat, "What is the fourth Harry Potter book?", "Harry Potter and the Goblet of Fire")
    add(hp_cat, "What is the fifth Harry Potter book?", "Harry Potter and the Order of the Phoenix")
    add(hp_cat, "What is the sixth Harry Potter book?", "Harry Potter and the Half-Blood Prince")
    add(hp_cat, "What is the seventh Harry Potter book?", "Harry Potter and the Deathly Hallows")
    add(hp_cat, "How many Harry Potter books are there?", "Seven")
    add(hp_cat, "Who wrote the Harry Potter series?", "J.K. Rowling")
    add(hp_cat, "What are all the Harry Potter books in order?", "Harry Potter and the Philosopher's Stone, Harry Potter and the Chamber of Secrets, Harry Potter and the Prisoner of Azkaban, Harry Potter and the Goblet of Fire, Harry Potter and the Order of the Phoenix, Harry Potter and the Half-Blood Prince, Harry Potter and the Deathly Hallows")
    add(hp_cat, "What is the first Harry Potter book called in the US?", "Harry Potter and the Sorcerer's Stone")
    add(hp_cat, "What Hogwarts house is Harry Potter in?", "Gryffindor")
    add(hp_cat, "What are the four Hogwarts houses?", "Gryffindor, Hufflepuff, Ravenclaw, Slytherin")
    add(hp_cat, "What is the name of Harry Potter's owl?", "Hedwig")
    add(hp_cat, "Who is the main villain in Harry Potter?", "Lord Voldemort")
    add(hp_cat, "What is Voldemort's real name?", "Tom Marvolo Riddle")
    add(hp_cat, "Who is the headmaster of Hogwarts in Harry Potter?", "Albus Dumbledore")
    add(hp_cat, "What is the sport played on broomsticks in Harry Potter?", "Quidditch")
    add(hp_cat, "What is Harry Potter's best friend's name?", "Ron Weasley")
    add(hp_cat, "What is Hermione Granger's full name?", "Hermione Jean Granger")
    add(hp_cat, "What position does Harry play in Quidditch?", "Seeker")
    add(hp_cat, "What is the Golden Snitch worth in Quidditch?", "150 points")
    add(hp_cat, "Who is the potions professor at Hogwarts?", "Severus Snape")
    add(hp_cat, "What is the name of the school for witchcraft and wizardry?", "Hogwarts School of Witchcraft and Wizardry")

    # ==========================================================================
    # 3. FIFTY SHADES OF GREY
    # ==========================================================================
    fs_cat = "fifty_shades"
    add(fs_cat, "What is the first Fifty Shades book?", "Fifty Shades of Grey")
    add(fs_cat, "What is the second Fifty Shades book?", "Fifty Shades Darker")
    add(fs_cat, "What is the third Fifty Shades book?", "Fifty Shades Freed")
    add(fs_cat, "What are the three Fifty Shades of Grey books?", "Fifty Shades of Grey, Fifty Shades Darker, Fifty Shades Freed")
    add(fs_cat, "Who wrote the Fifty Shades trilogy?", "E.L. James")
    add(fs_cat, "What are the main characters in Fifty Shades of Grey?", "Christian Grey and Anastasia Steele")
    add(fs_cat, "What is the name of the male lead in Fifty Shades of Grey?", "Christian Grey")

    # ==========================================================================
    # 4. TWILIGHT SERIES
    # ==========================================================================
    tw_cat = "twilight"
    add(tw_cat, "What is the first Twilight book?", "Twilight")
    add(tw_cat, "What is the second Twilight book?", "New Moon")
    add(tw_cat, "What is the third Twilight book?", "Eclipse")
    add(tw_cat, "What is the fourth Twilight book?", "Breaking Dawn")
    add(tw_cat, "What are all the Twilight books in order?", "Twilight, New Moon, Eclipse, Breaking Dawn")
    add(tw_cat, "Who wrote the Twilight series?", "Stephenie Meyer")
    add(tw_cat, "Who is the main character in Twilight?", "Bella Swan")
    add(tw_cat, "What is the vampire love interest's name in Twilight?", "Edward Cullen")
    add(tw_cat, "What is the werewolf's name in Twilight?", "Jacob Black")
    add(tw_cat, "What is the name of the vampire family in Twilight?", "The Cullen family")
    add(tw_cat, "Who does Bella marry in Twilight?", "Edward Cullen")
    add(tw_cat, "What is the name of Bella and Edward's daughter in Twilight?", "Renesmee Cullen")

    # ==========================================================================
    # 5. THE HUNGER GAMES
    # ==========================================================================
    hg_cat = "hunger_games"
    add(hg_cat, "What is the first Hunger Games book?", "The Hunger Games")
    add(hg_cat, "What is the second Hunger Games book?", "Catching Fire")
    add(hg_cat, "What is the third Hunger Games book?", "Mockingjay")
    add(hg_cat, "What are the Hunger Games books in order?", "The Hunger Games, Catching Fire, Mockingjay")
    add(hg_cat, "Who wrote the Hunger Games trilogy?", "Suzanne Collins")
    add(hg_cat, "Who is the main character in The Hunger Games?", "Katniss Everdeen")
    add(hg_cat, "Who is the male tribute from District 12 in The Hunger Games?", "Peeta Mellark")
    add(hg_cat, "What is the name of the dictator in The Hunger Games?", "President Snow")
    add(hg_cat, "What is the name of the country in The Hunger Games?", "Panem")
    add(hg_cat, "How many districts are there in Panem?", "13 (12 existing)")
    add(hg_cat, "What is Katniss's sister's name in The Hunger Games?", "Primrose Everdeen")

    # ==========================================================================
    # 6. LORD OF THE RINGS
    # ==========================================================================
    lotr_cat = "lord_of_the_rings"
    add(lotr_cat, "What is the first book in The Lord of the Rings?", "The Fellowship of the Ring")
    add(lotr_cat, "What is the second book in The Lord of the Rings?", "The Two Towers")
    add(lotr_cat, "What is the third book in The Lord of the Rings?", "The Return of the King")
    add(lotr_cat, "What are the Lord of the Rings books in order?", "The Fellowship of the Ring, The Two Towers, The Return of the King")
    add(lotr_cat, "What is the first Hobbit book?", "The Hobbit")
    add(lotr_cat, "Who wrote The Lord of the Rings and The Hobbit?", "J.R.R. Tolkien")
    add(lotr_cat, "Who is the main character in The Hobbit?", "Bilbo Baggins")
    add(lotr_cat, "Who is the main character in The Lord of the Rings?", "Frodo Baggins")
    add(lotr_cat, "What is the name of the ring in Lord of the Rings?", "The One Ring")
    add(lotr_cat, "Who is the Dark Lord in Lord of the Rings?", "Sauron")
    add(lotr_cat, "Who is the wizard in Lord of the Rings?", "Gandalf")
    add(lotr_cat, "What is the name of the elf in Lord of the Rings?", "Legolas")
    add(lotr_cat, "Who is the dwarf in Lord of the Rings?", "Gimli")
    add(lotr_cat, "Who is the ranger who becomes king in Lord of the Rings?", "Aragorn")
    add(lotr_cat, "What is the name of the world in Lord of the Rings?", "Middle-earth")

    # ==========================================================================
    # 7. GAME OF THRONES (Books)
    # ==========================================================================
    got_cat = "game_of_thrones"
    add(got_cat, "What is the first Game of Thrones book?", "A Game of Thrones")
    add(got_cat, "What is the second book in A Song of Ice and Fire?", "A Clash of Kings")
    add(got_cat, "What is the third book in A Song of Ice and Fire?", "A Storm of Swords")
    add(got_cat, "What is the fourth book in A Song of Ice and Fire?", "A Feast for Crows")
    add(got_cat, "What is the fifth book in A Song of Ice and Fire?", "A Dance with Dragons")
    add(got_cat, "What are the Game of Thrones books in order?", "A Game of Thrones, A Clash of Kings, A Storm of Swords, A Feast for Crows, A Dance with Dragons")
    add(got_cat, "How many books are planned for A Song of Ice and Fire?", "Seven")
    add(got_cat, "Who wrote the Game of Thrones series?", "George R.R. Martin")
    add(got_cat, "What is the name of the first book in A Song of Ice and Fire?", "A Game of Thrones")
    add(got_cat, "Who is the creator of the Game of Thrones TV series?", "David Benioff and D.B. Weiss")
    add(got_cat, "Which network aired Game of Thrones?", "HBO")
    add(got_cat, "Who sits on the Iron Throne at the end of Game of Thrones?", "Bran Stark")
    add(got_cat, "Who is the Mother of Dragons in Game of Thrones?", "Daenerys Targaryen")
    add(got_cat, "What is the sigil of House Stark?", "A direwolf")
    add(got_cat, "What is the sigil of House Lannister?", "A lion")
    add(got_cat, "What is the sigil of House Targaryen?", "A three-headed dragon")
    add(got_cat, "Who is the King in the North in Game of Thrones?", "Robb Stark")
    add(got_cat, "What is the Wall in Game of Thrones?", "A massive fortification of ice guarding the northern border of the Seven Kingdoms")
    add(got_cat, "Who are the protectors of the Wall in Game of Thrones?", "The Night's Watch")
    add(got_cat, "What is winter coming in Game of Thrones?", "A long winter and the return of the White Walkers")

    # ==========================================================================
    # 8. TV SHOWS — Friends
    # ==========================================================================
    tv_cat = "tv_shows"

    # Friends
    add(tv_cat, "What are the names of the six main characters in Friends?", "Rachel Green, Monica Geller, Phoebe Buffay, Joey Tribbiani, Chandler Bing, Ross Geller")
    add(tv_cat, "Who played Rachel Green on Friends?", "Jennifer Aniston")
    add(tv_cat, "Who played Monica Geller on Friends?", "Courteney Cox")
    add(tv_cat, "Who played Phoebe Buffay on Friends?", "Lisa Kudrow")
    add(tv_cat, "Who played Joey Tribbiani on Friends?", "Matt LeBlanc")
    add(tv_cat, "Who played Chandler Bing on Friends?", "Matthew Perry")
    add(tv_cat, "Who played Ross Geller on Friends?", "David Schwimmer")
    add(tv_cat, "How many seasons of Friends were there?", "10")
    add(tv_cat, "What is the name of the coffee shop in Friends?", "Central Perk")
    add(tv_cat, "What is Ross's catchphrase in Friends?", "We were on a break!")
    add(tv_cat, "What is Joey's catchphrase in Friends?", "How you doin'?")
    add(tv_cat, "Who does Ross end up with in Friends?", "Rachel Green")
    add(tv_cat, "Who does Chandler marry in Friends?", "Monica Geller")
    add(tv_cat, "What is Phoebe's twin sister's name in Friends?", "Ursula Buffay")
    add(tv_cat, "What is the name of Monica's apartment building in Friends?", "No specific name, but she lives at 90 Bedford Street, New York City")
    add(tv_cat, "What is the theme song of Friends called?", "I'll Be There for You by The Rembrandts")

    # The Office (US)
    add(tv_cat, "Who played Michael Scott in The Office?", "Steve Carell")
    add(tv_cat, "Who played Jim Halpert in The Office?", "John Krasinski")
    add(tv_cat, "Who played Pam Beesly in The Office?", "Jenna Fischer")
    add(tv_cat, "Who played Dwight Schrute in The Office?", "Rainn Wilson")
    add(tv_cat, "What is the name of the paper company in The Office?", "Dunder Mifflin")
    add(tv_cat, "What is Dwight's job title in The Office?", "Assistant to the Regional Manager")
    add(tv_cat, "What is the name of the town where The Office is set?", "Scranton, Pennsylvania")
    add(tv_cat, "How many seasons of The Office (US) are there?", "9")
    add(tv_cat, "Who does Jim marry in The Office?", "Pam Beesly")
    add(tv_cat, "What is the name of the HR representative in The Office?", "Toby Flenderson")
    add(tv_cat, "What is Michael Scott's catchphrase in The Office?", "That's what she said")
    add(tv_cat, "What is the beet farm called in The Office?", "Schrute Farms")

    # That 70s Show
    add(tv_cat, "Who are the main characters in That 70s Show?", "Eric Forman, Donna Pinciotti, Steven Hyde, Michael Kelso, Jackie Burkhart, Fez")
    add(tv_cat, "Who played Eric Forman in That 70s Show?", "Topher Grace")
    add(tv_cat, "Who played Donna Pinciotti in That 70s Show?", "Laura Prepon")
    add(tv_cat, "Who played Steven Hyde in That 70s Show?", "Danny Masterson")
    add(tv_cat, "Who played Michael Kelso in That 70s Show?", "Ashton Kutcher")
    add(tv_cat, "Who played Jackie Burkhart in That 70s Show?", "Mila Kunis")
    add(tv_cat, "Who played Fez in That 70s Show?", "Wilmer Valderrama")
    add(tv_cat, "Who does Eric end up with in That 70s Show?", "Donna Pinciotti")
    add(tv_cat, "Who does Eric Forman marry in That 70s Show?", "Donna Pinciotti")
    add(tv_cat, "How many seasons of That 70s Show are there?", "8")
    add(tv_cat, "What is the name of the Forman family's basement hangout?", "The basement (it's the Formans' basement)")
    add(tv_cat, "What year is That 70s Show set in?", "1976-1979")
    add(tv_cat, "What is the name of the car Eric drives in That 70s Show?", "A 1970 Oldsmobile Vista Cruiser")
    add(tv_cat, "Who is Red Forman in That 70s Show?", "Eric's father, a strict WWII veteran")
    add(tv_cat, "What is Red Forman's famous threat in That 70s Show?", "I'm gonna put my foot in your ass")
    add(tv_cat, "Where does That 70s Show take place?", "Point Place, Wisconsin")

    # Breaking Bad
    add(tv_cat, "Who played Walter White in Breaking Bad?", "Bryan Cranston")
    add(tv_cat, "Who played Jesse Pinkman in Breaking Bad?", "Aaron Paul")
    add(tv_cat, "What is Walter White's alias in Breaking Bad?", "Heisenberg")
    add(tv_cat, "What is the blue meth called in Breaking Bad?", "Blue Sky")
    add(tv_cat, "How many seasons of Breaking Bad are there?", "5")
    add(tv_cat, "Where is Breaking Bad set?", "Albuquerque, New Mexico")
    add(tv_cat, "What is the spin-off show from Breaking Bad called?", "Better Call Saul")
    add(tv_cat, "Who plays Saul Goodman in Breaking Bad?", "Bob Odenkirk")
    add(tv_cat, "What is Walter's wife's name in Breaking Bad?", "Skyler White")
    add(tv_cat, "What is Walter White's profession before cooking meth?", "High school chemistry teacher")
    add(tv_cat, "What disease does Walter White have in Breaking Bad?", "Lung cancer")
    add(tv_cat, "What is the name of the fast-food chain in Breaking Bad?", "Los Pollos Hermanos")
    add(tv_cat, "Who is the owner of Los Pollos Hermanos in Breaking Bad?", "Gustavo Fring")
    add(tv_cat, "What is the RV called in Breaking Bad?", "The Crystal Ship")
    add(tv_cat, "Who is the DEA agent in Breaking Bad?", "Hank Schrader")

    # Stranger Things
    add(tv_cat, "Who are the main kids in Stranger Things?", "Mike Wheeler, Eleven, Lucas Sinclair, Dustin Henderson, Will Byers")
    add(tv_cat, "Who played Eleven in Stranger Things?", "Millie Bobby Brown")
    add(tv_cat, "Where is Stranger Things set?", "Hawkins, Indiana")
    add(tv_cat, "What is the alternate dimension called in Stranger Things?", "The Upside Down")
    add(tv_cat, "What is the monster called in Stranger Things season 1?", "The Demogorgon")
    add(tv_cat, "What are Eleven's psychic powers in Stranger Things?", "Telekinesis and telepathy")
    add(tv_cat, "What is the government lab called in Stranger Things?", "Hawkins National Laboratory")
    add(tv_cat, "How many seasons of Stranger Things are there?", "4 (as of 2024)")
    add(tv_cat, "Who created Stranger Things?", "The Duffer Brothers")
    add(tv_cat, "What is Eleven's real name in Stranger Things?", "Jane Hopper (born Jane Ives)")
    add(tv_cat, "What is the name of the older teen group in Stranger Things?", "Nancy Wheeler, Jonathan Byers, Steve Harrington")
    add(tv_cat, "What is Steve Harrington's job in later Stranger Things seasons?", "Working at the video store Family Video")

    # Seinfeld
    add(tv_cat, "Who are the four main characters in Seinfeld?", "Jerry Seinfeld, George Costanza, Elaine Benes, Cosmo Kramer")
    add(tv_cat, "Who played Jerry Seinfeld on Seinfeld?", "Jerry Seinfeld")
    add(tv_cat, "Who played George Costanza on Seinfeld?", "Jason Alexander")
    add(tv_cat, "Who played Elaine Benes on Seinfeld?", "Julia Louis-Dreyfus")
    add(tv_cat, "Who played Cosmo Kramer on Seinfeld?", "Michael Richards")
    add(tv_cat, "How many seasons of Seinfeld are there?", "9")
    add(tv_cat, "What is George Costanza's catchphrase in Seinfeld?", "It's not a lie if you believe it")
    add(tv_cat, "What is Kramer's catchphrase in Seinfeld?", "I'm out there!")
    add(tv_cat, "What is the name of the diner in Seinfeld?", "Monk's Cafe")
    add(tv_cat, "What is Jerry's neighbor's name in Seinfeld?", "Cosmo Kramer")
    add(tv_cat, "What is the show about on Seinfeld?", "A show about nothing")
    add(tv_cat, "What is Newman's first name in Seinfeld?", "It is never revealed (only referred to as Newman)")

    # The Simpsons
    add(tv_cat, "What are the names of the main Simpson family members?", "Homer, Marge, Bart, Lisa, Maggie")
    add(tv_cat, "Who is the voice of Homer Simpson?", "Dan Castellaneta")
    add(tv_cat, "Who is the voice of Marge Simpson?", "Julie Kavner")
    add(tv_cat, "Who is the voice of Bart Simpson?", "Nancy Cartwright")
    add(tv_cat, "Who is the voice of Lisa Simpson?", "Yeardley Smith")
    add(tv_cat, "Where does The Simpsons take place?", "Springfield")
    add(tv_cat, "What is Homer's favorite beer in The Simpsons?", "Duff Beer")
    add(tv_cat, "What is the name of the bar in The Simpsons?", "Moe's Tavern")
    add(tv_cat, "Who is the bartender at Moe's Tavern in The Simpsons?", "Moe Szyslak")
    add(tv_cat, "What is the name of the nuclear power plant owner in The Simpsons?", "Mr. Charles Montgomery Burns")
    add(tv_cat, "What is Homer's catchphrase in The Simpsons?", "D'oh!")
    add(tv_cat, "What is Bart's catchphrase in The Simpsons?", "Eat my shorts!")
    add(tv_cat, "How many seasons of The Simpsons are there?", "35+ (currently airing)")
    add(tv_cat, "Who created The Simpsons?", "Matt Groening")
    add(tv_cat, "What is the name of the Simpsons' next-door neighbor?", "Ned Flanders")
    add(tv_cat, "What is the name of the comic book store owner in The Simpsons?", "Comic Book Guy (Jeff Albertson)")
    add(tv_cat, "What is the name of Springfield's rival town?", "Shelbyville")

    # ==========================================================================
    # 9. SPORTS — NFL
    # ==========================================================================
    sports_cat = "sports"

    # NFL Teams
    nfl_teams = [
        "Arizona Cardinals", "Atlanta Falcons", "Baltimore Ravens", "Buffalo Bills",
        "Carolina Panthers", "Chicago Bears", "Cincinnati Bengals", "Cleveland Browns",
        "Dallas Cowboys", "Denver Broncos", "Detroit Lions", "Green Bay Packers",
        "Houston Texans", "Indianapolis Colts", "Jacksonville Jaguars", "Kansas City Chiefs",
        "Las Vegas Raiders", "Los Angeles Chargers", "Los Angeles Rams", "Miami Dolphins",
        "Minnesota Vikings", "New England Patriots", "New Orleans Saints", "New York Giants",
        "New York Jets", "Philadelphia Eagles", "Pittsburgh Steelers", "San Francisco 49ers",
        "Seattle Seahawks", "Tampa Bay Buccaneers", "Tennessee Titans", "Washington Commanders"
    ]
    for t in nfl_teams:
        add(sports_cat, f"What NFL team is based in {t.split()[-1]}?", t)
    add(sports_cat, "How many teams are in the NFL?", "32")
    add(sports_cat, "How many teams are in the NFC?", "16")
    add(sports_cat, "How many teams are in the AFC?", "16")
    add(sports_cat, "What are all 32 NFL teams?", "Arizona Cardinals, Atlanta Falcons, Baltimore Ravens, Buffalo Bills, Carolina Panthers, Chicago Bears, Cincinnati Bengals, Cleveland Browns, Dallas Cowboys, Denver Broncos, Detroit Lions, Green Bay Packers, Houston Texans, Indianapolis Colts, Jacksonville Jaguars, Kansas City Chiefs, Las Vegas Raiders, Los Angeles Chargers, Los Angeles Rams, Miami Dolphins, Minnesota Vikings, New England Patriots, New Orleans Saints, New York Giants, New York Jets, Philadelphia Eagles, Pittsburgh Steelers, San Francisco 49ers, Seattle Seahawks, Tampa Bay Buccaneers, Tennessee Titans, Washington Commanders")

    # Super Bowl winners (last 20 years: Super Bowls XXXIX — LVIII, 2004-2023 seasons)
    sb_winners = [
        ("XXXIX", "2004", "New England Patriots"),
        ("XL", "2005", "Pittsburgh Steelers"),
        ("XLI", "2006", "Indianapolis Colts"),
        ("XLII", "2007", "New York Giants"),
        ("XLIII", "2008", "Pittsburgh Steelers"),
        ("XLIV", "2009", "New Orleans Saints"),
        ("XLV", "2010", "Green Bay Packers"),
        ("XLVI", "2011", "New York Giants"),
        ("XLVII", "2012", "Baltimore Ravens"),
        ("XLVIII", "2013", "Seattle Seahawks"),
        ("XLIX", "2014", "New England Patriots"),
        ("50", "2015", "Denver Broncos"),
        ("LI", "2016", "New England Patriots"),
        ("LII", "2017", "Philadelphia Eagles"),
        ("LIII", "2018", "New England Patriots"),
        ("LIV", "2019", "Kansas City Chiefs"),
        ("LV", "2020", "Tampa Bay Buccaneers"),
        ("LVI", "2021", "Los Angeles Rams"),
        ("LVII", "2022", "Kansas City Chiefs"),
        ("LVIII", "2023", "Kansas City Chiefs"),
    ]
    for roman, year, team in sb_winners:
        add(sports_cat, f"Who won Super Bowl {roman}?", team)
        add(sports_cat, f"Who won the Super Bowl in {year}?", team)
        add(sports_cat, f"Who was the {year} Super Bowl champion?", team)

    add(sports_cat, "Who won Super Bowl LVIII?", "Kansas City Chiefs")
    add(sports_cat, "Who won the most recent Super Bowl as of 2024?", "Kansas City Chiefs (Super Bowl LVIII, 2023 season)")
    add(sports_cat, "Which team has the most Super Bowl wins?", "Pittsburgh Steelers and New England Patriots (6 each)")

    # Vikings specific
    add(sports_cat, "When is the last time the Minnesota Vikings were in the Super Bowl?", "Super Bowl XI (1976 season, played January 1977)")
    add(sports_cat, "When is the last time the Vikings were in the NFC Championship?", "The 2017 season (January 2018), losing to the Philadelphia Eagles")
    add(sports_cat, "When is the last time the Vikings were in the NFC Championship game?", "The 2017 season (January 2018), losing to the Philadelphia Eagles")
    add(sports_cat, "Have the Vikings ever won the Super Bowl?", "No, the Minnesota Vikings have never won the Super Bowl")
    add(sports_cat, "How many Super Bowls have the Vikings lost?", "4 (Super Bowls IV, VIII, IX, XI)")
    add(sports_cat, "When did the Vikings last appear in the NFC Championship?", "2017 season (January 2018)")
    add(sports_cat, "What is the Minnesota Vikings' all-time record in Super Bowls?", "0-4 (lost Super Bowls IV, VIII, IX, XI)")
    add(sports_cat, "Which team has the most Super Bowl appearances?", "New England Patriots (11 appearances)")

    # Famous NFL players
    famous_nfl = [
        ("Tom Brady", "quarterback", "New England Patriots, Tampa Bay Buccaneers", "7 Super Bowl wins, most all-time"),
        ("Jerry Rice", "wide receiver", "San Francisco 49ers, Oakland Raiders, Seattle Seahawks", "all-time receiving yards leader"),
        ("Peyton Manning", "quarterback", "Indianapolis Colts, Denver Broncos", "2 Super Bowl wins, 5 MVP awards"),
        ("Jim Brown", "running back", "Cleveland Browns", "considered one of greatest RBs ever"),
        ("Lawrence Taylor", "linebacker", "New York Giants", "revolutionized the linebacker position"),
        ("Joe Montana", "quarterback", "San Francisco 49ers, Kansas City Chiefs", "4 Super Bowl wins, no interceptions in Super Bowls"),
        ("Walter Payton", "running back", "Chicago Bears", "all-time rushing leader at retirement"),
        ("Barry Sanders", "running back", "Detroit Lions", "one of the most elusive runners in NFL history"),
        ("Reggie White", "defensive end", "Philadelphia Eagles, Green Bay Packers", "all-time sacks leader at retirement"),
        ("Deion Sanders", "cornerback", "multiple teams", "only player to play in both a Super Bowl and World Series"),
        ("Patrick Mahomes", "quarterback", "Kansas City Chiefs", "3 Super Bowl wins, known for no-look passes"),
        ("JJ Watt", "defensive end", "Houston Texans, Arizona Cardinals", "3-time Defensive Player of the Year"),
        ("Aaron Rodgers", "quarterback", "Green Bay Packers, New York Jets", "4 MVP awards, Super Bowl XLV champion"),
        ("John Elway", "quarterback", "Denver Broncos", "2 Super Bowl wins, 5 AFC championships"),
        ("Brett Favre", "quarterback", "Green Bay Packers, New York Jets, Minnesota Vikings", "3 MVP awards, Super Bowl XXXI champion"),
    ]
    for name, pos, teams, notable in famous_nfl:
        add(sports_cat, f"Who is {name} in the NFL?", f"{name} is a former {pos} who played for {teams}, known for {notable}")
        add(sports_cat, f"What position did {name} play in the NFL?", pos)
    add(sports_cat, "Who is the all-time leading rusher in NFL history?", "Emmitt Smith")
    add(sports_cat, "Who is the all-time leading passer in NFL history?", "Tom Brady (rushing and TD leader also Brady)")

    # NBA teams
    nba_teams = [
        "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets",
        "Chicago Bulls", "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets",
        "Detroit Pistons", "Golden State Warriors", "Houston Rockets", "Indiana Pacers",
        "Los Angeles Clippers", "Los Angeles Lakers", "Memphis Grizzlies", "Miami Heat",
        "Milwaukee Bucks", "Minnesota Timberwolves", "New Orleans Pelicans", "New York Knicks",
        "Oklahoma City Thunder", "Orlando Magic", "Philadelphia 76ers", "Phoenix Suns",
        "Portland Trail Blazers", "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors",
        "Utah Jazz", "Washington Wizards"
    ]
    add(sports_cat, "How many teams are in the NBA?", "30")
    for t in nba_teams[:10]:
        add(sports_cat, f"What NBA team is from {t.split()[-1]}?", t)

    # NBA champions recent
    nba_champs = [
        ("2023-24", "Boston Celtics"),
        ("2022-23", "Denver Nuggets"),
        ("2021-22", "Golden State Warriors"),
        ("2020-21", "Milwaukee Bucks"),
        ("2019-20", "Los Angeles Lakers"),
        ("2018-19", "Toronto Raptors"),
        ("2017-18", "Golden State Warriors"),
        ("2016-17", "Golden State Warriors"),
        ("2015-16", "Cleveland Cavaliers"),
        ("2014-15", "Golden State Warriors"),
    ]
    for season, team in nba_champs:
        add(sports_cat, f"Who won the NBA championship in {season}?", team)

    # Famous basketball players
    add(sports_cat, "Who is considered the greatest basketball player of all time?", "Michael Jordan (often debated with LeBron James)")
    add(sports_cat, "How many NBA championships did Michael Jordan win?", "6")
    add(sports_cat, "Who is the all-time leading scorer in NBA history?", "LeBron James")
    add(sports_cat, "How many NBA championships has LeBron James won?", "4")
    add(sports_cat, "Who is the all-time leading scorer for a single game in NBA?", "Wilt Chamberlain (100 points)")
    add(sports_cat, "Who has the most NBA championships all-time?", "Bill Russell (11 championships)")
    add(sports_cat, "Which NBA team has the most championships?", "Boston Celtics (18)")
    add(sports_cat, "Who are the members of the Dream Team?", "Michael Jordan, Magic Johnson, Larry Bird, Charles Barkley, Patrick Ewing, David Robinson, Karl Malone, John Stockton, Chris Mullin, Clyde Drexler, Scottie Pippen, Christian Laettner")
    add(sports_cat, "Who won the NBA MVP award in 2023-24?", "Nikola Jokic")
    add(sports_cat, "Who is known as King James in the NBA?", "LeBron James")
    add(sports_cat, "Who is known as the Black Mamba in the NBA?", "Kobe Bryant")
    add(sports_cat, "Who is known as the Greek Freak in the NBA?", "Giannis Antetokounmpo")
    add(sports_cat, "Who is known as Chef Curry in the NBA?", "Stephen Curry")
    add(sports_cat, "Who has the most 3-pointers in NBA history?", "Stephen Curry")

    # ==========================================================================
    # 10. MOVIES
    # ==========================================================================
    movie_cat = "movies"

    # Marvel Cinematic Universe
    mcq = movie_cat
    add(mcq, "What was the first Marvel Cinematic Universe movie?", "Iron Man (2008)")
    add(mcq, "What is the highest-grossing Marvel movie?", "Avengers: Endgame")
    add(mcq, "Who played Iron Man in the MCU?", "Robert Downey Jr.")
    add(mcq, "Who played Captain America in the MCU?", "Chris Evans")
    add(mcq, "Who played Thor in the MCU?", "Chris Hemsworth")
    add(mcq, "Who played Black Widow in the MCU?", "Scarlett Johansson")
    add(mcq, "Who played the Hulk in the MCU?", "Mark Ruffalo")
    add(mcq, "Who played Thanos in the MCU?", "Josh Brolin")
    add(mcq, "What is the Infinity Gauntlet in the MCU?", "A device that holds the six Infinity Stones")
    add(mcq, "What are the six Infinity Stones in the MCU?", "Space, Mind, Reality, Power, Time, Soul")
    add(mcq, "What is the name of the third Avengers movie?", "Avengers: Infinity War")
    add(mcq, "What is the name of the fourth Avengers movie?", "Avengers: Endgame")
    add(mcq, "Who is the main villain in Avengers: Infinity War?", "Thanos")
    add(mcq, "What is the Snap called in Avengers: Infinity War?", "The Blip or The Snap")
    add(mcq, "Who directed the first Iron Man movie?", "Jon Favreau")
    add(mcq, "What is the name of Wakanda's king in Black Panther?", "T'Challa")
    add(mcq, "Who played Black Panther in the MCU?", "Chadwick Boseman")
    add(mcq, "What is the fictional metal in Wakanda called?", "Vibranium")
    add(mcq, "Who is Spider-Man in the MCU?", "Tom Holland")
    add(mcq, "What is the name of the magical art in Doctor Strange?", "The Mystic Arts")
    add(mcq, "Who played Doctor Strange in the MCU?", "Benedict Cumberbatch")
    add(mcq, "Who is the god of mischief in the MCU?", "Loki")
    add(mcq, "Who played Loki in the MCU?", "Tom Hiddleston")
    add(mcq, "What are the Avengers?", "A team of superheroes formed by Nick Fury to protect Earth")
    add(mcq, "Who is the director of S.H.I.E.L.D. in the MCU?", "Nick Fury")
    add(mcq, "Who played Nick Fury in the MCU?", "Samuel L. Jackson")
    add(mcq, "What is the name of the Guardians of the Galaxy team leader?", "Star-Lord (Peter Quill)")
    add(mcq, "Who plays Star-Lord in the MCU?", "Chris Pratt")
    add(mcq, "How many MCU movies are there as of 2024?", "33")
    add(mcq, "What is the first Avengers movie called?", "The Avengers (2012)")
    add(mcq, "Who is the protagonist of Captain Marvel?", "Carol Danvers")
    add(mcq, "Who plays Captain Marvel in the MCU?", "Brie Larson")
    add(mcq, "What is the name of the Nova Corps in the MCU?", "Nova Corps (from Xandar)")
    add(mcq, "Who made the Infinity Gauntlet in the comics?", "Eitri the Dwarf King")

    # Star Wars
    add(mcq, "What is the first Star Wars movie released?", "Star Wars: Episode IV - A New Hope (1977)")
    add(mcq, "What is the original Star Wars trilogy?", "A New Hope, The Empire Strikes Back, Return of the Jedi")
    add(mcq, "What is the prequel Star Wars trilogy?", "The Phantom Menace, Attack of the Clones, Revenge of the Sith")
    add(mcq, "What is the sequel Star Wars trilogy?", "The Force Awakens, The Last Jedi, The Rise of Skywalker")
    add(mcq, "Who created Star Wars?", "George Lucas")
    add(mcq, "Who is the main villain of the original Star Wars trilogy?", "Darth Vader")
    add(mcq, "Who is Anakin Skywalker in Star Wars?", "The Chosen One who becomes Darth Vader")
    add(mcq, "Who directed Star Wars: A New Hope?", "George Lucas")
    add(mcq, "Who directed Star Wars: The Empire Strikes Back?", "Irvin Kershner")
    add(mcq, "Who played Luke Skywalker?", "Mark Hamill")
    add(mcq, "Who played Princess Leia?", "Carrie Fisher")
    add(mcq, "Who played Han Solo?", "Harrison Ford")
    add(mcq, "What is the name of Han Solo's ship?", "Millennium Falcon")
    add(mcq, "What is the name of Luke's home planet in Star Wars?", "Tatooine")
    add(mcq, "What is the weapon of a Jedi?", "Lightsaber")
    add(mcq, "What is the name of the evil empire in Star Wars?", "The Galactic Empire")
    add(mcq, "Who is the Emperor in Star Wars?", "Emperor Palpatine (Darth Sidious)")
    add(mcq, "What is the Force in Star Wars?", "A mystical energy field that connects all living things in the universe")
    add(mcq, "Who was Anakin Skywalker's master?", "Obi-Wan Kenobi")
    add(mcq, "Who played Obi-Wan Kenobi?", "Alec Guinness (original), Ewan McGregor (prequel)")
    add(mcq, "What is the name of the Rebel base on Hoth?", "Echo Base")
    add(mcq, "What is the name of the bounty hunter in Star Wars?", "Boba Fett")
    add(mcq, "Who is Yoda in Star Wars?", "A wise Jedi Master who trained Luke")
    add(mcq, "What is Mando's real name in The Mandalorian?", "Din Djarin")
    add(mcq, "What is Grogu in The Mandalorian?", "The Child (nicknamed Baby Yoda)")
    add(mcq, "Who created The Mandalorian?", "Jon Favreau")
    add(mcq, "What is the name of the droid in Star Wars?", "R2-D2")
    add(mcq, "What is the protocol droid's name in Star Wars?", "C-3PO")
    add(mcq, "What is Kylo Ren's real name in Star Wars?", "Ben Solo")
    add(mcq, "Who plays Kylo Ren?", "Adam Driver")
    add(mcq, "What is the name of the cantina in Mos Eisley?", "Chalmun's Spaceport Cantina")

    # Top-grossing films
    add(mcq, "What is the highest-grossing film of all time?", "Avatar (2009)")
    add(mcq, "What is the second highest-grossing film of all time?", "Avengers: Endgame (2019)")
    add(mcq, "What is the third highest-grossing film of all time?", "Avatar: The Way of Water (2022)")
    add(mcq, "What is the fourth highest-grossing film of all time?", "Titanic (1997)")
    add(mcq, "What is the fifth highest-grossing film of all time?", "Star Wars: The Force Awakens (2015)")
    add(mcq, "Who directed Avatar?", "James Cameron")
    add(mcq, "Who directed Titanic?", "James Cameron")
    add(mcq, "Who directed the first Jurassic Park?", "Steven Spielberg")
    add(mcq, "Who directed Jaws?", "Steven Spielberg")
    add(mcq, "Who directed Pulp Fiction?", "Quentin Tarantino")
    add(mcq, "Who directed The Dark Knight?", "Christopher Nolan")
    add(mcq, "Who directed Inception?", "Christopher Nolan")
    add(mcq, "Who directed Schindler's List?", "Steven Spielberg")
    add(mcq, "Who directed The Godfather?", "Francis Ford Coppola")
    add(mcq, "Who directed E.T. the Extra-Terrestrial?", "Steven Spielberg")
    add(mcq, "What is the highest-grossing animated film of all time?", "Frozen II (2019)")
    add(mcq, "What is the highest-grossing R-rated film of all time?", "Joker (2019)")
    add(mcq, "Who directed Joker?", "Todd Phillips")
    add(mcq, "Who played the Joker in The Dark Knight?", "Heath Ledger")
    add(mcq, "Who played the Joker in Joker (2019)?", "Joaquin Phoenix")
    add(mcq, "What is the name of James Bond's code number?", "007")
    add(mcq, "Who created James Bond?", "Ian Fleming")
    add(mcq, "Who is the voice of Woody in Toy Story?", "Tom Hanks")
    add(mcq, "Who is the voice of Buzz Lightyear in Toy Story?", "Tim Allen")
    add(mcq, "Who directed Toy Story?", "John Lasseter")
    add(mcq, "What was the first Pixar movie?", "Toy Story (1995)")
    add(mcq, "Who directed The Matrix?", "The Wachowskis")

    # ==========================================================================
    # 11. MUSIC
    # ==========================================================================
    music_cat = "music"
    add(music_cat, "Who is known as the King of Pop?", "Michael Jackson")
    add(music_cat, "Who is known as the King of Rock and Roll?", "Elvis Presley")
    add(music_cat, "Who is known as the Queen of Pop?", "Madonna")
    add(music_cat, "Who is known as the Queen of Soul?", "Aretha Franklin")
    add(music_cat, "Who is known as the King of Rock?", "Chuck Berry")
    add(music_cat, "Who is the best-selling music artist of all time?", "The Beatles")
    add(music_cat, "Who is the best-selling solo artist of all time?", "Elvis Presley")
    add(music_cat, "Who are the members of The Beatles?", "John Lennon, Paul McCartney, George Harrison, Ringo Starr")
    add(music_cat, "Who is the best-selling female artist of all time?", "Madonna")
    add(music_cat, "What is the best-selling album of all time?", "Thriller by Michael Jackson")
    add(music_cat, "How many Grammy Awards has Beyonce won?", "32 (most of any artist)")
    add(music_cat, "How many Grammy Awards has Taylor Swift won?", "14")
    add(music_cat, "Who has won the most Grammy Awards?", "Beyonce (32)")
    add(music_cat, "Who is the leader of the band Queen?", "Freddie Mercury")
    add(music_cat, "What are the names of the members of Queen?", "Freddie Mercury, Brian May, John Deacon, Roger Taylor")
    add(music_cat, "Who is the lead singer of Nirvana?", "Kurt Cobain")
    add(music_cat, "Who is the lead singer of U2?", "Bono")
    add(music_cat, "Who is the lead singer of The Rolling Stones?", "Mick Jagger")
    add(music_cat, "Who is the lead singer of Led Zeppelin?", "Robert Plant")
    add(music_cat, "What band released the album Abbey Road?", "The Beatles")
    add(music_cat, "What band released the album The Dark Side of the Moon?", "Pink Floyd")
    add(music_cat, "What band released the album Back in Black?", "AC/DC")
    add(music_cat, "Who sang Bohemian Rhapsody?", "Queen")
    add(music_cat, "Who sang Like a Rolling Stone?", "Bob Dylan")
    add(music_cat, "Who sang Respect?", "Aretha Franklin")
    add(music_cat, "Who sang Smells Like Teen Spirit?", "Nirvana")
    add(music_cat, "Who sang Billie Jean?", "Michael Jackson")
    add(music_cat, "Who sang Like a Virgin?", "Madonna")
    add(music_cat, "Who sang I Will Always Love You?", "Whitney Houston")
    add(music_cat, "Who sang Rolling in the Deep?", "Adele")
    add(music_cat, "Who sang Bad Guy?", "Billie Eilish")
    add(music_cat, "Who sang Shake It Off?", "Taylor Swift")
    add(music_cat, "Who sang Blinding Lights?", "The Weeknd")
    add(music_cat, "Who sang Shape of You?", "Ed Sheeran")
    add(music_cat, "Who sang Uptown Funk?", "Mark Ronson featuring Bruno Mars")
    add(music_cat, "Who sang Happy?", "Pharrell Williams")
    add(music_cat, "Who sang Old Town Road?", "Lil Nas X")
    add(music_cat, "Who sang Despacito?", "Luis Fonsi featuring Daddy Yankee")
    add(music_cat, "Who sang Someone Like You?", "Adele")
    add(music_cat, "Who sang Umbrella?", "Rihanna")
    add(music_cat, "Who sang Single Ladies?", "Beyonce")
    add(music_cat, "Who is the best-selling female artist of the 21st century?", "Taylor Swift")
    add(music_cat, "Which rapper has the most Grammy Awards?", "Jay-Z (24)")
    add(music_cat, "Who sang Lose Yourself?", "Eminem")
    add(music_cat, "Who sang God's Plan?", "Drake")
    add(music_cat, "Who sang HUMBLE?", "Kendrick Lamar")
    add(music_cat, "What is the best-selling album by a female artist?", "21 by Adele")
    add(music_cat, "What is the best-selling album of the 21st century?", "21 by Adele")
    add(music_cat, "Who is the most-streamed artist on Spotify?", "Drake")
    add(music_cat, "Who has the most number-one hits on the Billboard Hot 100?", "The Beatles (20)")

    # ==========================================================================
    # 12. SCIENCE
    # ==========================================================================
    sci_cat = "science"
    add(sci_cat, "What is the largest planet in our solar system?", "Jupiter")
    add(sci_cat, "What is the smallest planet in our solar system?", "Mercury")
    add(sci_cat, "What is the hottest planet in our solar system?", "Venus")
    add(sci_cat, "What is the coldest planet in our solar system?", "Neptune")
    add(sci_cat, "What is the farthest planet from the Sun?", "Neptune")
    add(sci_cat, "What is the closest planet to the Sun?", "Mercury")
    add(sci_cat, "What are the planets in order from the Sun?", "Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune")
    add(sci_cat, "How many planets are in the solar system?", "8")
    add(sci_cat, "What is the largest moon of Saturn?", "Titan")
    add(sci_cat, "What is the largest moon of Jupiter?", "Ganymede")
    add(sci_cat, "What is the Earth's largest moon called?", "The Moon (Luna)")
    add(sci_cat, "What is the chemical symbol for water?", "H2O")
    add(sci_cat, "What is the chemical symbol for carbon dioxide?", "CO2")
    add(sci_cat, "What is the chemical symbol for oxygen gas?", "O2")
    add(sci_cat, "What is the chemical symbol for gold?", "Au")
    add(sci_cat, "What is the chemical symbol for silver?", "Ag")
    add(sci_cat, "What is the chemical symbol for iron?", "Fe")
    add(sci_cat, "What is the chemical symbol for sodium?", "Na")
    add(sci_cat, "What is the chemical symbol for potassium?", "K")
    add(sci_cat, "What is the chemical symbol for chlorine?", "Cl")
    add(sci_cat, "What is the chemical symbol for calcium?", "Ca")
    add(sci_cat, "What is the chemical symbol for lead?", "Pb")
    add(sci_cat, "What is the chemical symbol for mercury?", "Hg")
    add(sci_cat, "What is the lightest element?", "Hydrogen")
    add(sci_cat, "What is the heaviest naturally occurring element?", "Uranium")
    add(sci_cat, "What is the most abundant element in the universe?", "Hydrogen")
    add(sci_cat, "What is the most abundant element in the Earth's crust?", "Oxygen")
    add(sci_cat, "What is the most abundant element in the Earth's atmosphere?", "Nitrogen")
    add(sci_cat, "What is the atomic number of carbon?", "6")
    add(sci_cat, "What is the atomic number of hydrogen?", "1")
    add(sci_cat, "What is the atomic number of oxygen?", "8")
    add(sci_cat, "What is the atomic number of gold?", "79")
    add(sci_cat, "What is the atomic number of iron?", "26")
    add(sci_cat, "What is the largest organ in the human body?", "The skin")
    add(sci_cat, "What is the smallest bone in the human body?", "The stapes (in the middle ear)")
    add(sci_cat, "What is the largest bone in the human body?", "The femur (thigh bone)")
    add(sci_cat, "What is the strongest muscle in the human body?", "The masseter (jaw muscle)")
    add(sci_cat, "How many bones are in the adult human body?", "206")
    add(sci_cat, "How many teeth does an adult human have?", "32")
    add(sci_cat, "What is the average human body temperature in Fahrenheit?", "98.6°F")
    add(sci_cat, "What is the average human body temperature in Celsius?", "37°C")
    add(sci_cat, "How many chromosomes do humans have?", "46 (23 pairs)")
    add(sci_cat, "What blood type is the universal donor?", "O negative")
    add(sci_cat, "What blood type is the universal recipient?", "AB positive")
    add(sci_cat, "What is the largest organ inside the human body?", "The liver")
    add(sci_cat, "What is the largest artery in the human body?", "The aorta")
    add(sci_cat, "What is the formula for the speed of light?", "299,792,458 meters per second")
    add(sci_cat, "What is the speed of light?", "299,792,458 m/s (approximately 300,000 km/s)")
    add(sci_cat, "What is the name of our galaxy?", "The Milky Way")
    add(sci_cat, "What is the closest star to Earth?", "The Sun")
    add(sci_cat, "What is the closest star to Earth besides the Sun?", "Proxima Centauri")
    add(sci_cat, "How many moons does Earth have?", "1")
    add(sci_cat, "How many moons does Mars have?", "2 (Phobos and Deimos)")
    add(sci_cat, "How many moons does Jupiter have?", "95 known moons")
    add(sci_cat, "How many moons does Saturn have?", "146 known moons")
    add(sci_cat, "Which scientist developed the theory of general relativity?", "Albert Einstein")
    add(sci_cat, "Who developed the theory of evolution by natural selection?", "Charles Darwin")
    add(sci_cat, "Who is known as the father of modern physics?", "Albert Einstein")
    add(sci_cat, "Who is known as the father of modern chemistry?", "Antoine Lavoisier")
    add(sci_cat, "Who is known as the father of modern biology?", "Charles Darwin")
    add(sci_cat, "Who discovered penicillin?", "Alexander Fleming")
    add(sci_cat, "Who discovered the structure of DNA?", "James Watson and Francis Crick")
    add(sci_cat, "What does DNA stand for?", "Deoxyribonucleic acid")
    add(sci_cat, "What does RNA stand for?", "Ribonucleic acid")
    add(sci_cat, "What is the powerhouse of the cell?", "The mitochondria")
    add(sci_cat, "What is the largest animal on Earth?", "The blue whale")
    add(sci_cat, "What is the fastest land animal?", "The cheetah")
    add(sci_cat, "What is the largest land animal?", "The African elephant")
    add(sci_cat, "What is the tallest animal on Earth?", "The giraffe")
    add(sci_cat, "What is the most intelligent animal after humans?", "Great apes (chimpanzees, bonobos)")
    add(sci_cat, "How many species of penguins are there?", "18")
    add(sci_cat, "What is the boiling point of water in Celsius?", "100°C")
    add(sci_cat, "What is the freezing point of water in Celsius?", "0°C")
    add(sci_cat, "What is the boiling point of water in Fahrenheit?", "212°F")
    add(sci_cat, "What is the freezing point of water in Fahrenheit?", "32°F")

    # ==========================================================================
    # 13. GEOGRAPHY
    # ==========================================================================
    geo_cat = "geography"

    # World capitals (beyond existing common knowledge, add major ones)
    world_capitals = [
        ("China", "Beijing"), ("Japan", "Tokyo"), ("India", "New Delhi"),
        ("South Korea", "Seoul"), ("Indonesia", "Jakarta"), ("Philippines", "Manila"),
        ("Vietnam", "Hanoi"), ("Thailand", "Bangkok"), ("Turkey", "Ankara"),
        ("Saudi Arabia", "Riyadh"), ("Iran", "Tehran"), ("Iraq", "Baghdad"),
        ("Israel", "Jerusalem"), ("Egypt", "Cairo"), ("South Africa", "Pretoria"),
        ("Nigeria", "Abuja"), ("Kenya", "Nairobi"), ("Ethiopia", "Addis Ababa"),
        ("Morocco", "Rabat"), ("Algeria", "Algiers"), ("Argentina", "Buenos Aires"),
        ("Chile", "Santiago"), ("Colombia", "Bogota"), ("Peru", "Lima"),
        ("Mexico", "Mexico City"), ("Cuba", "Havana"), ("Jamaica", "Kingston"),
        ("Norway", "Oslo"), ("Sweden", "Stockholm"), ("Finland", "Helsinki"),
        ("Denmark", "Copenhagen"), ("Netherlands", "Amsterdam"), ("Belgium", "Brussels"),
        ("Switzerland", "Bern"), ("Austria", "Vienna"), ("Poland", "Warsaw"),
        ("Russia", "Moscow"), ("Ukraine", "Kyiv"), ("Greece", "Athens"),
        ("Portugal", "Lisbon"), ("Ireland", "Dublin"), ("Scotland", "Edinburgh"),
        ("Australia", "Canberra"), ("New Zealand", "Wellington"),
        ("United Arab Emirates", "Abu Dhabi"), ("Qatar", "Doha"),
        ("Singapore", "Singapore"), ("Malaysia", "Kuala Lumpur"),
    ]
    for country, capital in world_capitals:
        add(geo_cat, f"What is the capital of {country}?", capital)

    # US States and capitals
    us_state_capitals = [
        ("Alabama", "Montgomery"), ("Alaska", "Juneau"), ("Arizona", "Phoenix"),
        ("Arkansas", "Little Rock"), ("California", "Sacramento"), ("Colorado", "Denver"),
        ("Connecticut", "Hartford"), ("Delaware", "Dover"), ("Florida", "Tallahassee"),
        ("Georgia", "Atlanta"), ("Hawaii", "Honolulu"), ("Idaho", "Boise"),
        ("Illinois", "Springfield"), ("Indiana", "Indianapolis"), ("Iowa", "Des Moines"),
        ("Kansas", "Topeka"), ("Kentucky", "Frankfort"), ("Louisiana", "Baton Rouge"),
        ("Maine", "Augusta"), ("Maryland", "Annapolis"), ("Massachusetts", "Boston"),
        ("Michigan", "Lansing"), ("Minnesota", "Saint Paul"), ("Mississippi", "Jackson"),
        ("Missouri", "Jefferson City"), ("Montana", "Helena"), ("Nebraska", "Lincoln"),
        ("Nevada", "Carson City"), ("New Hampshire", "Concord"), ("New Jersey", "Trenton"),
        ("New Mexico", "Santa Fe"), ("New York", "Albany"), ("North Carolina", "Raleigh"),
        ("North Dakota", "Bismarck"), ("Ohio", "Columbus"), ("Oklahoma", "Oklahoma City"),
        ("Oregon", "Salem"), ("Pennsylvania", "Harrisburg"), ("Rhode Island", "Providence"),
        ("South Carolina", "Columbia"), ("South Dakota", "Pierre"), ("Tennessee", "Nashville"),
        ("Texas", "Austin"), ("Utah", "Salt Lake City"), ("Vermont", "Montpelier"),
        ("Virginia", "Richmond"), ("Washington", "Olympia"), ("West Virginia", "Charleston"),
        ("Wisconsin", "Madison"), ("Wyoming", "Cheyenne"),
    ]
    for state, capital in us_state_capitals:
        add(geo_cat, f"What is the capital of {state}?", capital)
    add(geo_cat, "How many states are in the United States?", "50")
    add(geo_cat, "What is the largest US state by area?", "Alaska")
    add(geo_cat, "What is the smallest US state by area?", "Rhode Island")
    add(geo_cat, "What is the largest US state by population?", "California")
    add(geo_cat, "What is the smallest US state by population?", "Wyoming")

    # World landmarks
    world_landmarks = [
        ("The tallest mountain in the world", "Mount Everest (29,032 feet)"),
        ("The longest river in the world", "The Nile River"),
        ("The largest ocean in the world", "The Pacific Ocean"),
        ("The smallest ocean in the world", "The Arctic Ocean"),
        ("The largest continent by area", "Asia"),
        ("The smallest continent by area", "Australia"),
        ("The largest desert in the world", "The Antarctic Desert"),
        ("The largest hot desert in the world", "The Sahara Desert"),
        ("The deepest point in the ocean", "The Mariana Trench (Challenger Deep)"),
        ("The highest waterfall in the world", "Angel Falls (Venezuela)"),
        ("The largest lake in the world", "The Caspian Sea"),
        ("The deepest lake in the world", "Lake Baikal (Siberia)"),
        ("The largest country by area", "Russia"),
        ("The smallest country by area", "Vatican City"),
        ("The most populous country", "India"),
        ("The most populous city in the world", "Tokyo, Japan"),
        ("Where is the Great Wall of China located?", "China"),
        ("Where is the Eiffel Tower located?", "Paris, France"),
        ("Where is the Statue of Liberty located?", "New York City, USA"),
        ("Where is the Colosseum located?", "Rome, Italy"),
        ("Where is the Taj Mahal located?", "Agra, India"),
        ("Where is Machu Picchu located?", "Peru"),
        ("Where is the Great Pyramid of Giza located?", "Giza, Egypt"),
        ("Where is the Sydney Opera House located?", "Sydney, Australia"),
        ("Where is the Leaning Tower of Pisa located?", "Pisa, Italy"),
        ("Where is Christ the Redeemer statue located?", "Rio de Janeiro, Brazil"),
        ("Where is Stonehenge located?", "Wiltshire, England"),
        ("Where is the Acropolis located?", "Athens, Greece"),
        ("What is the name of the river that flows through London?", "The River Thames"),
        ("What is the name of the river that flows through Paris?", "The River Seine"),
        ("What is the name of the river that flows through Rome?", "The Tiber River"),
        ("What is the name of the river that flows through Cairo?", "The Nile River"),
    ]
    for q, a in world_landmarks:
        add(geo_cat, q, a)

    add(geo_cat, "What is the official language of Brazil?", "Portuguese")
    add(geo_cat, "What is the official language of Argentina?", "Spanish")
    add(geo_cat, "What is the official language of Canada?", "English and French")
    add(geo_cat, "What is the official language of Switzerland?", "German, French, Italian, Romansh")
    add(geo_cat, "How many time zones does Russia have?", "11")
    add(geo_cat, "How many time zones does the US have?", "9 (including territories)")
    add(geo_cat, "What is the longest border in the world?", "The US-Canada border")
    add(geo_cat, "What is the smallest country in the world by area?", "Vatican City")
    add(geo_cat, "What is the second smallest country in the world?", "Monaco")

    # ==========================================================================
    # 14. HISTORY
    # ==========================================================================
    hist_cat = "history"

    # US Presidents
    us_presidents = [
        (1, "George Washington", "1789-1797"),
        (2, "John Adams", "1797-1801"),
        (3, "Thomas Jefferson", "1801-1809"),
        (4, "James Madison", "1809-1817"),
        (5, "James Monroe", "1817-1825"),
        (6, "John Quincy Adams", "1825-1829"),
        (7, "Andrew Jackson", "1829-1837"),
        (8, "Martin Van Buren", "1837-1841"),
        (9, "William Henry Harrison", "1841 (one month)"),
        (10, "John Tyler", "1841-1845"),
        (11, "James K. Polk", "1845-1849"),
        (12, "Zachary Taylor", "1849-1850"),
        (13, "Millard Fillmore", "1850-1853"),
        (14, "Franklin Pierce", "1853-1857"),
        (15, "James Buchanan", "1857-1861"),
        (16, "Abraham Lincoln", "1861-1865"),
        (17, "Andrew Johnson", "1865-1869"),
        (18, "Ulysses S. Grant", "1869-1877"),
        (19, "Rutherford B. Hayes", "1877-1881"),
        (20, "James A. Garfield", "1881"),
        (21, "Chester A. Arthur", "1881-1885"),
        (22, "Grover Cleveland", "1885-1889"),
        (23, "Benjamin Harrison", "1889-1893"),
        (24, "Grover Cleveland", "1893-1897"),
        (25, "William McKinley", "1897-1901"),
        (26, "Theodore Roosevelt", "1901-1909"),
        (27, "William Howard Taft", "1909-1913"),
        (28, "Woodrow Wilson", "1913-1921"),
        (29, "Warren G. Harding", "1921-1923"),
        (30, "Calvin Coolidge", "1923-1929"),
        (31, "Herbert Hoover", "1929-1933"),
        (32, "Franklin D. Roosevelt", "1933-1945"),
        (33, "Harry S. Truman", "1945-1953"),
        (34, "Dwight D. Eisenhower", "1953-1961"),
        (35, "John F. Kennedy", "1961-1963"),
        (36, "Lyndon B. Johnson", "1963-1969"),
        (37, "Richard Nixon", "1969-1974"),
        (38, "Gerald Ford", "1974-1977"),
        (39, "Jimmy Carter", "1977-1981"),
        (40, "Ronald Reagan", "1981-1989"),
        (41, "George H.W. Bush", "1989-1993"),
        (42, "Bill Clinton", "1993-2001"),
        (43, "George W. Bush", "2001-2009"),
        (44, "Barack Obama", "2009-2017"),
        (45, "Donald Trump", "2017-2021"),
        (46, "Joe Biden", "2021-present"),
    ]
    for num, name, years in us_presidents:
        add(hist_cat, f"Who was the {num}th president of the United States?", name)
        add(hist_cat, f"Who was the US president number {num}?", name)
        add(hist_cat, f"When did {name} serve as president?", years)

    top_10_presidents = [
        "George Washington", "Thomas Jefferson", "Abraham Lincoln",
        "Theodore Roosevelt", "Franklin D. Roosevelt", "Harry S. Truman",
        "Dwight D. Eisenhower", "John F. Kennedy", "Ronald Reagan", "Barack Obama"
    ]
    for p in top_10_presidents:
        add(hist_cat, f"What number president was {p}?", f"{[n for n, name, _ in us_presidents if name == p][0]}th")

    # Major wars and dates
    add(hist_cat, "When did World War I start?", "1914")
    add(hist_cat, "When did World War I end?", "1918")
    add(hist_cat, "When did World War II start?", "1939")
    add(hist_cat, "When did World War II end?", "1945")
    add(hist_cat, "When did the American Civil War start?", "1861")
    add(hist_cat, "When did the American Civil War end?", "1865")
    add(hist_cat, "When did the American Revolutionary War start?", "1775")
    add(hist_cat, "When did the American Revolutionary War end?", "1783")
    add(hist_cat, "When was the Declaration of Independence signed?", "1776")
    add(hist_cat, "When did the attack on Pearl Harbor occur?", "December 7, 1941")
    add(hist_cat, "When did the Berlin Wall fall?", "November 9, 1989")
    add(hist_cat, "When was the United Nations founded?", "1945")
    add(hist_cat, "When was NATO founded?", "1949")
    add(hist_cat, "When did the Soviet Union collapse?", "1991")
    add(hist_cat, "When was the first moon landing?", "July 20, 1969")
    add(hist_cat, "Who was the first person to walk on the moon?", "Neil Armstrong")
    add(hist_cat, "Who was the second person to walk on the moon?", "Buzz Aldrin")
    add(hist_cat, "What was the name of the Apollo 11 mission?", "Apollo 11")
    add(hist_cat, "What was the name of the first American in space?", "Alan Shepard")
    add(hist_cat, "Who was the first American woman in space?", "Sally Ride")
    add(hist_cat, "What was the name of the first artificial satellite?", "Sputnik 1")
    add(hist_cat, "Who was the first person in space?", "Yuri Gagarin")
    add(hist_cat, "When was the Titanic disaster?", "April 15, 1912")
    add(hist_cat, "When did the French Revolution begin?", "1789")
    add(hist_cat, "Who was the first Emperor of Rome?", "Augustus (Octavian)")
    add(hist_cat, "Who was the first Emperor of China?", "Qin Shi Huang")
    add(hist_cat, "When was the Magna Carta signed?", "1215")
    add(hist_cat, "When did Christopher Columbus discover the Americas?", "1492")
    add(hist_cat, "When was the Renaissance period?", "14th to 17th century")
    add(hist_cat, "When was the Industrial Revolution?", "1760 to 1840")

    # ==========================================================================
    # 15. ADDITIONAL CURRENT EVENTS / GENERAL KNOWLEDGE
    # ==========================================================================
    gen_cat = "general_knowledge"
    add(gen_cat, "What is the name of the current Pope?", "Pope Francis")
    add(gen_cat, "What is the name of the current King of England?", "King Charles III")
    add(gen_cat, "Who was Queen of England before Charles III?", "Queen Elizabeth II")
    add(gen_cat, "How long did Queen Elizabeth II reign?", "70 years (1952-2022)")
    add(gen_cat, "What is the currency of the United States?", "The US dollar (USD)")
    add(gen_cat, "What is the currency of Japan?", "The Japanese yen (JPY)")
    add(gen_cat, "What is the currency of the United Kingdom?", "The British pound sterling (GBP)")
    add(gen_cat, "What is the currency of the European Union?", "The euro (EUR)")
    add(gen_cat, "What is the currency of China?", "The Chinese yuan (CNY)")
    add(gen_cat, "What is the currency of India?", "The Indian rupee (INR)")
    add(gen_cat, "What is the currency of Australia?", "The Australian dollar (AUD)")
    add(gen_cat, "What is the currency of Canada?", "The Canadian dollar (CAD)")
    add(gen_cat, "What is the currency of Brazil?", "The Brazilian real (BRL)")
    add(gen_cat, "What is the currency of Switzerland?", "The Swiss franc (CHF)")
    add(gen_cat, "What is the currency of Russia?", "The Russian ruble (RUB)")
    add(gen_cat, "What is the currency of Mexico?", "The Mexican peso (MXN)")
    add(gen_cat, "What is the currency of South Korea?", "The South Korean won (KRW)")
    add(gen_cat, "What is the tallest building in the world?", "Burj Khalifa in Dubai")
    add(gen_cat, "What is the largest airport in the world?", "Hartsfield-Jackson Atlanta International Airport (by passenger traffic)")
    add(gen_cat, "What is the busiest airport in the world?", "Hartsfield-Jackson Atlanta International Airport")
    add(gen_cat, "What is the busiest port in the world?", "Port of Shanghai")
    add(gen_cat, "What is the most spoken language in the world?", "Mandarin Chinese (by native speakers)")
    add(gen_cat, "What is the most spoken language in the world by total speakers?", "English")
    add(gen_cat, "What is the most populous city in the United States?", "New York City")
    add(gen_cat, "What is the second most populous city in the United States?", "Los Angeles")
    add(gen_cat, "What is the third most populous city in the United States?", "Chicago")
    add(gen_cat, "What is the largest state in the US by area?", "Alaska")
    add(gen_cat, "What is the most common blood type?", "O positive")
    add(gen_cat, "What is the rarest blood type?", "AB negative")
    add(gen_cat, "What is the longest river in the United States?", "The Missouri River")
    add(gen_cat, "What is the largest lake in the United States?", "Lake Superior")
    add(gen_cat, "What is the deepest lake in the United States?", "Crater Lake")
    add(gen_cat, "What is the tallest mountain in the United States?", "Denali (Mount McKinley) in Alaska")
    add(gen_cat, "What is the largest national park in the US?", "Wrangell-St. Elias National Park (Alaska)")

    # ==========================================================================
    # 16. ADDITIONAL BOOK SERIES FACTS
    # ==========================================================================
    book_cat = "books"
    add(book_cat, "Who wrote the Percy Jackson series?", "Rick Riordan")
    add(book_cat, "What is the first Percy Jackson book?", "Percy Jackson and the Olympians: The Lightning Thief")
    add(book_cat, "Who wrote the Chronicles of Narnia?", "C.S. Lewis")
    add(book_cat, "What is the first Chronicles of Narnia book?", "The Lion, the Witch and the Wardrobe")
    add(book_cat, "Who wrote The Da Vinci Code?", "Dan Brown")
    add(book_cat, "Who wrote The Girl with the Dragon Tattoo?", "Stieg Larsson")
    add(book_cat, "Who wrote To Kill a Mockingbird?", "Harper Lee")
    add(book_cat, "Who wrote 1984?", "George Orwell")
    add(book_cat, "Who wrote Pride and Prejudice?", "Jane Austen")
    add(book_cat, "Who wrote The Great Gatsby?", "F. Scott Fitzgerald")
    add(book_cat, "Who wrote Moby Dick?", "Herman Melville")
    add(book_cat, "Who wrote Frankenstein?", "Mary Shelley")
    add(book_cat, "Who wrote The Catcher in the Rye?", "J.D. Salinger")
    add(book_cat, "Who wrote The Adventures of Huckleberry Finn?", "Mark Twain")
    add(book_cat, "Who wrote War and Peace?", "Leo Tolstoy")
    add(book_cat, "Who wrote The Hobbit?", "J.R.R. Tolkien")
    add(book_cat, "Who wrote The Divine Comedy?", "Dante Alighieri")
    add(book_cat, "Who wrote Don Quixote?", "Miguel de Cervantes")
    add(book_cat, "Who wrote One Hundred Years of Solitude?", "Gabriel Garcia Marquez")
    add(book_cat, "Who wrote The Hitchhiker's Guide to the Galaxy?", "Douglas Adams")

    # ==========================================================================
    # 17. MORE SCIENCE: ELEMENTS
    # ==========================================================================
    elements = [
        ("Hydrogen", "H", 1, "the lightest element"),
        ("Helium", "He", 2, "a noble gas used in balloons"),
        ("Lithium", "Li", 3, "a light metal used in batteries"),
        ("Carbon", "C", 6, "the basis of organic chemistry"),
        ("Nitrogen", "N", 7, "the most abundant gas in Earth's atmosphere"),
        ("Oxygen", "O", 8, "essential for respiration"),
        ("Fluorine", "F", 9, "the most reactive element"),
        ("Neon", "Ne", 10, "a noble gas used in signs"),
        ("Sodium", "Na", 11, "a reactive alkali metal"),
        ("Aluminum", "Al", 13, "a lightweight metal used in cans and aircraft"),
        ("Silicon", "Si", 14, "used in computer chips"),
        ("Chlorine", "Cl", 17, "used for water purification"),
        ("Potassium", "K", 19, "essential for plant growth"),
        ("Calcium", "Ca", 20, "essential for bones and teeth"),
        ("Iron", "Fe", 26, "the most common element on Earth by mass"),
        ("Copper", "Cu", 29, "a reddish metal used in electrical wiring"),
        ("Zinc", "Zn", 30, "used in galvanizing and batteries"),
        ("Silver", "Ag", 47, "a precious metal with the highest electrical conductivity"),
        ("Gold", "Au", 79, "a precious metal that does not tarnish"),
        ("Mercury", "Hg", 80, "the only liquid metal at room temperature"),
        ("Lead", "Pb", 82, "a heavy metal historically used in pipes"),
        ("Uranium", "U", 92, "used in nuclear power and weapons"),
    ]
    for name, sym, num, desc in elements:
        add(sci_cat, f"What is the chemical symbol for {name}?", sym)
        add(sci_cat, f"What element has the symbol {sym}?", name)
        add(sci_cat, f"What is the atomic number of {name}?", str(num))
    add(sci_cat, "How many known elements are there?", "118")
    add(sci_cat, "How many naturally occurring elements are there?", "92")

    # ==========================================================================
    # 18. MORE TV SHOW ENDING FACTS
    # ==========================================================================
    add(tv_cat, "How does Breaking Bad end?", "Walter White dies in a meth lab after saving Jesse Pinkman")
    add(tv_cat, "Who survives at the end of Game of Thrones?", "Bran Stark becomes king, Sansa rules the North, Arya sails west, Jon Snow goes north")
    add(tv_cat, "Who dies in the Red Wedding in Game of Thrones?", "Robb Stark, Catelyn Stark, and many Stark bannermen")
    add(tv_cat, "What is the Battle of Winterfell in Game of Thrones?", "The battle between the living and the Army of the Dead")
    add(tv_cat, "Who kills the Night King in Game of Thrones?", "Arya Stark")
    add(tv_cat, "What is the Wall in Game of Thrones made of?", "Ice")
    add(tv_cat, "Who is the Three-Eyed Raven in Game of Thrones?", "Bran Stark")

    # ==========================================================================
    # 19. MORE SPORTS — MLB, NHL, SOCCER
    # ==========================================================================
    # Soccer
    add(sports_cat, "Which country has won the most FIFA World Cups?", "Brazil (5)")
    add(sports_cat, "Who won the 2022 FIFA World Cup?", "Argentina")
    add(sports_cat, "Who won the 2018 FIFA World Cup?", "France")
    add(sports_cat, "Who won the 2014 FIFA World Cup?", "Germany")
    add(sports_cat, "Who is considered the greatest soccer player of all time?", "Lionel Messi (often debated with Cristiano Ronaldo)")
    add(sports_cat, "Who has won the most Ballon d'Or awards?", "Lionel Messi (8)")
    add(sports_cat, "Who plays for Inter Miami in Major League Soccer?", "Lionel Messi")
    add(sports_cat, "Which club has won the most UEFA Champions League titles?", "Real Madrid (15)")
    add(sports_cat, "Who is known as CR7 in soccer?", "Cristiano Ronaldo")
    add(sports_cat, "What is the name of the English Premier League?", "The Premier League")
    add(sports_cat, "Which club has won the most English Premier League titles?", "Manchester United (20, including First Division era)")
    add(sports_cat, "Who is the all-time leading scorer in FIFA World Cup history?", "Miroslav Klose (16 goals)")
    add(sports_cat, "What country hosts the 2026 FIFA World Cup?", "United States, Canada, and Mexico")

    # ==========================================================================
    # WRITE OUTPUT
    # ==========================================================================
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for fact in facts:
            f.write(json.dumps(fact) + "\n")
    print(f"Wrote {len(facts)} facts to {OUTPUT_PATH}")


if __name__ == "__main__":
    build_facts()
