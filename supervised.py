import tinker
import numpy as np

# No arguments needed; it will use the TINKER_API_KEY from your environment
service_client = tinker.ServiceClient()

#discover mdoels 
print("Available models:")
for item in service_client.get_server_capabilities().supported_models:
    print("- " + item.model_name)
from tinker_cookbook.hyperparam_utils import get_lora_param_count
 
from tinker_cookbook.hyperparam_utils import get_lora_lr_over_full_finetune_lr
 
model_name = "Qwen/Qwen3-30B-A3B-Base"
print(get_lora_lr_over_full_finetune_lr(model_name))

#prime lora-  I will have 494,088,192 trainable parameters added on top of the frozen base model.
model_name = "Qwen/Qwen3-30B-A3B-Base"
print(get_lora_param_count(model_name, lora_rank=32))

base_model = "Qwen/Qwen3-30B-A3B-Base"
training_client = service_client.create_lora_training_client(
    base_model=base_model,
    rank=32, #hp
   
)

#create dataset - about new york for supervised learning 
examples = [
    {"input": "What is the tallest building in New York City?", 
     "output": "One World Trade Center."},
    {"input": "Where can I see the Statue of Liberty?", 
     "output": "You can visit Liberty Island to see the Statue of Liberty."},
    {"input": "What are some famous museums in NYC?", 
     "output": "The Metropolitan Museum of Art, MoMA, and the American Museum of Natural History."},
    {"input": "How do I get from Times Square to Central Park?", 
     "output": "You can walk north along 7th Avenue or take the subway (N, Q, R, W to 59th Street – Columbus Circle)."},
    {"input": "What is a must-visit landmark in Manhattan?", 
     "output": "The Empire State Building is a must-visit landmark in Manhattan."},
    {"input": "Where should I eat classic New York pizza?", 
     "output": "Try Joe's Pizza in Greenwich Village or Di Fara Pizza in Brooklyn."},
    {"input": "How can I get from Brooklyn to Manhattan?", 
     "output": "You can take the subway, drive, or use the East River ferries."},
    {"input": "What is the largest park in Manhattan?", 
     "output": "Central Park."},
    {"input": "Where can I see Broadway shows?", 
     "output": "Broadway shows are performed in the Theater District around Times Square."},
    {"input": "What are some famous bridges in NYC?", 
     "output": "Brooklyn Bridge, Manhattan Bridge, Williamsburg Bridge, and Queensboro Bridge."},
    {"input": "Where can I find street art in NYC?", 
     "output": "Check out Bushwick in Brooklyn or the Lower East Side in Manhattan."},
    {"input": "What is a famous observation deck?", 
     "output": "Top of the Rock at Rockefeller Center or One World Observatory."},
    {"input": "Which neighborhood is known for hipster culture?", 
     "output": "Williamsburg in Brooklyn is known for its hipster vibe."},
    {"input": "Where is the financial district?", 
     "output": "The Financial District is in Lower Manhattan, including Wall Street."},
    {"input": "Where can I go shopping in NYC?", 
     "output": "Fifth Avenue, SoHo, and Times Square are popular shopping areas."},
    {"input": "What is the best way to see the NYC skyline?", 
     "output": "Take a boat tour around Manhattan or visit an observation deck."},
    {"input": "Where is the High Line?", 
     "output": "The High Line is an elevated park on Manhattan's West Side."},
    {"input": "Where can I try famous bagels?", 
     "output": "Try Ess-a-Bagel, Russ & Daughters, or Absolute Bagels."},
    {"input": "Where is Chinatown located?", 
     "output": "Chinatown is in Lower Manhattan."},
    {"input": "What is the main train station in NYC?", 
     "output": "Grand Central Terminal in Midtown Manhattan."},
    {"input": "Where can I see modern art?", 
     "output": "The Museum of Modern Art (MoMA) in Midtown Manhattan."},
    {"input": "What is a famous food market?", 
     "output": "Chelsea Market in Manhattan."},
    {"input": "Where can I see Times Square lights?", 
     "output": "Times Square itself is the place to see the famous lights and billboards."},
    {"input": "What is the main airport for international flights?", 
     "output": "John F. Kennedy International Airport (JFK)."},
    {"input": "Where is the Brooklyn Botanic Garden?", 
     "output": "In the borough of Brooklyn, near Prospect Park."},
    {"input": "What is the famous green space in Brooklyn?", 
     "output": "Prospect Park."},
    {"input": "Where can I see the Empire State Building lights?", 
     "output": "From the building’s observation deck or nearby streets in Midtown Manhattan."},
    {"input": "What is the subway system called?", 
     "output": "The NYC Subway."},
    {"input": "Where can I find jazz clubs?", 
     "output": "Village Vanguard and Blue Note in Greenwich Village."},
    {"input": "Where is the Flatiron Building?", 
     "output": "At the intersection of Fifth Avenue and Broadway in Manhattan."},
    {"input": "Where is the United Nations headquarters?", 
     "output": "In Midtown East, Manhattan."},
    {"input": "Where can I visit historic Ellis Island?", 
     "output": "Ellis Island Immigration Museum, accessible by ferry from Battery Park."},
    {"input": "What is the Bronx Zoo?", 
     "output": "A large zoological park located in the Bronx."},
    {"input": "Where can I find Rockefeller Center?", 
     "output": "Midtown Manhattan, between 48th and 51st Streets."},
    {"input": "Where is the New York Public Library main branch?", 
     "output": "On Fifth Avenue at 42nd Street, Manhattan."},
    {"input": "Where can I see street performers?", 
     "output": "Times Square and Washington Square Park are popular spots."},
    {"input": "What is the Metropolitan Opera?", 
     "output": "A world-renowned opera company at Lincoln Center."},
    {"input": "Where can I ride the Staten Island Ferry?", 
     "output": "The Staten Island Ferry departs from the southern tip of Manhattan at Whitehall Terminal."},
    {"input": "Where is the Chrysler Building?", 
     "output": "Midtown Manhattan, near Grand Central Terminal."},
    {"input": "What is Yankee Stadium?", 
     "output": "Home stadium of the New York Yankees in the Bronx."},
    {"input": "Where can I see Wall Street?", 
     "output": "In the Financial District of Lower Manhattan."},
    {"input": "Where is Madison Square Garden?", 
     "output": "Above Penn Station in Midtown Manhattan."},
    {"input": "Where can I go ice skating in winter?", 
     "output": "Rockefeller Center or Bryant Park ice rinks."},
    {"input": "What is the Guggenheim Museum?", 
     "output": "An art museum on the Upper East Side of Manhattan, famous for its spiral architecture."},
    {"input": "Where is the New York Botanical Garden?", 
     "output": "In the Bronx, near the Bronx Zoo."},
    {"input": "Where is Coney Island?", 
     "output": "A neighborhood in Brooklyn, famous for its amusement parks and boardwalk."},
    {"input": "Where can I see street art murals?", 
     "output": "Bushwick in Brooklyn is a popular area for murals and street art."},
    {"input": "Where is the MoMA PS1 museum?", 
     "output": "In Long Island City, Queens."},
    {"input": "Where can I watch a baseball game?", 
     "output": "Yankee Stadium in the Bronx or Citi Field in Queens."},
    {"input": "What is the Apollo Theater?", 
     "output": "A historic music hall in Harlem, Manhattan."},
    {"input": "Where is the Intrepid Sea, Air & Space Museum?", 
     "output": "On the west side of Midtown Manhattan, on the aircraft carrier USS Intrepid."},
    {"input": "Where can I visit Brooklyn Heights Promenade?", 
     "output": "In Brooklyn, with views of the Manhattan skyline and Brooklyn Bridge."},
    {"input": "What is the New York Transit Museum?", 
     "output": "A museum in Brooklyn showcasing the history of NYC transit."},
    {"input": "Where is Battery Park?", 
     "output": "At the southern tip of Manhattan, near ferries to Staten Island and the Statue of Liberty."},
    {"input": "Where can I see contemporary art?", 
     "output": "The Whitney Museum of American Art in the Meatpacking District."},
    {"input": "Where is Lincoln Center?", 
     "output": "On the Upper West Side of Manhattan, home to opera, ballet, and orchestras."},
    {"input": "Where can I ride a historic carousel?", 
     "output": "Central Park or Coney Island have historic carousels."},
    {"input": "Where is the Tenement Museum?", 
     "output": "On the Lower East Side, Manhattan, showcasing immigrant history."},
    {"input": "Where can I see the Manhattan skyline from a rooftop?", 
     "output": "Top of the Rock, One World Observatory, or rooftop bars in Midtown."},
    {"input": "Where is SoHo?", 
     "output": "In Lower Manhattan, known for shopping and art galleries."},
    {"input": "Where is Little Italy?", 
     "output": "In Lower Manhattan, adjacent to Chinatown."},
    {"input": "Where is Washington Square Park?", 
     "output": "In Greenwich Village, Manhattan."},
    {"input": "Where can I see the New York City Marathon?", 
     "output": "The marathon runs through all five boroughs, starting in Staten Island and finishing in Central Park."},
    {"input": "Where is the New York Hall of Science?", 
     "output": "In Flushing Meadows–Corona Park, Queens."},
    {"input": "Where is Flushing Meadows Corona Park?", 
     "output": "In Queens, home to the US Open tennis tournament."},
    {"input": "Where is Roosevelt Island?", 
     "output": "In the East River, accessible by tram or subway from Manhattan."},
    {"input": "Where is the New York Stock Exchange?", 
     "output": "In the Financial District, Lower Manhattan."},
    {"input": "Where is the South Street Seaport?", 
     "output": "In Lower Manhattan along the East River."},
    {"input": "Where can I go kayaking in NYC?", 
     "output": "The Hudson River, Brooklyn Bridge Park, and the East River offer kayaking programs."},
    {"input": "Where is the Frick Collection?", 
     "output": "On the Upper East Side of Manhattan, featuring European art."},
    {"input": "Where is the New York Aquarium?", 
     "output": "In Coney Island, Brooklyn."},
    {"input": "Where can I see live theater besides Broadway?", 
     "output": "Off-Broadway theaters in Manhattan or Brooklyn Academy of Music (BAM)."},
    {"input": "Where is Bryant Park?", 
     "output": "Between 40th and 42nd Streets, Midtown Manhattan."},
    {"input": "Where is the Cloisters museum?", 
     "output": "In Fort Tryon Park, Upper Manhattan, specializing in medieval European art."},
    {"input": "Where can I take ferry rides?", 
     "output": "Staten Island Ferry, NYC Ferry routes along East and Harlem Rivers."},
    {"input": "Where is Tribeca?", 
     "output": "In Lower Manhattan, known for the Tribeca Film Festival."},
    {"input": "Where is the Lower East Side?", 
     "output": "In Manhattan, east of SoHo and north of Chinatown."},
    {"input": "Where can I find Chinatown restaurants?", 
     "output": "In Manhattan’s Chinatown, especially along Canal Street and Mott Street."},
    {"input": "Where is the Meatpacking District?", 
     "output": "In Manhattan, near Chelsea, known for nightlife and galleries."},
    {"input": "Where is Hell's Kitchen?", 
     "output": "In Midtown Manhattan, west of Times Square."},
    {"input": "Where is Upper East Side?", 
     "output": "Upper Manhattan, east of Central Park."},
    {"input": "Where is Upper West Side?", 
     "output": "Upper Manhattan, west of Central Park."},
    {"input": "Where is Midtown Manhattan?", 
     "output": "Between 34th and 59th Streets, including Times Square and Grand Central Terminal."},
    {"input": "Where is Downtown Manhattan?", 
     "output": "Lower Manhattan, including Wall Street, Battery Park, and SoHo."},
    {"input": "Where is Harlem?", 
     "output": "In Upper Manhattan, north of Central Park."},
    {"input": "Where is Queens?", 
     "output": "Queens is a borough of NYC, east of Manhattan and Brooklyn."},
    {"input": "Where is Staten Island?", 
     "output": "Staten Island is the southernmost borough of NYC, accessible by ferry from Manhattan."},
    {"input": "Where is the Bronx?", 
     "output": "The Bronx is the northernmost borough of NYC."},
    {"input": "Where is Brooklyn?", 
     "output": "Brooklyn is a borough south of Manhattan, across the East River."},
    {"input": "Where is Long Island City?", 
     "output": "In Queens, along the East River waterfront."},
    {"input": "Where is Flushing?", 
     "output": "In Queens, home to Flushing Meadows–Corona Park and Citi Field."},
    {"input": "Where is Astoria?", 
     "output": "In Queens, known for food and cultural diversity."},
    {"input": "Where is Greenwich Village?", 
     "output": "In Lower Manhattan, known for arts and nightlife."},
    {"input": "Where is Chelsea?", 
     "output": "In Manhattan, known for galleries and the High Line."},
    {"input": "Where is Soho?", 
     "output": "In Lower Manhattan, famous for shopping and art."},
    {"input": "Where is Tribeca?", 
     "output": "In Lower Manhattan, known for film festivals and lofts."},
    {"input": "Where is the Financial District?", 
     "output": "In Lower Manhattan, home to Wall Street and NYSE."},
    {"input": "Where is the Theater District?", 
     "output": "In Midtown Manhattan, around Times Square."},
    {"input": "Where is Hudson Yards?", 
     "output": "On the west side of Midtown Manhattan, featuring The Vessel."},
    {"input": "Where is the Flatiron District?", 
     "output": "In Manhattan, around the Flatiron Building."},
    {"input": "Where is Union Square?", 
     "output": "In Manhattan, at the intersection of Broadway and Fourth Avenue."},
    {"input": "Where is Madison Square Park?", 
     "output": "In Manhattan, near the Flatiron Building."},
    {"input": "Where is Battery Park?", 
     "output": "At the southern tip of Manhattan."},
    {"input": "Where is the East Village?", 
     "output": "In Manhattan, east of Greenwich Village."},
    {"input": "Where is the West Village?", 
     "output": "In Manhattan, west of Greenwich Village."},
    {"input": "Where is the Upper East Side?", 
     "output": "In Manhattan, east of Central Park."},
    {"input": "Where is the Upper West Side?", 
     "output": "In Manhattan, west of Central Park."},
    {"input": "Where is Roosevelt Island?", 
     "output": "In the East River, between Manhattan and Queens."},
    {"input": "Where is Governors Island?", 
     "output": "South of Manhattan, accessible by ferry."},
    {"input": "Where is Liberty Island?", 
     "output": "In New York Harbor, home to the Statue of Liberty."},
    {"input": "Where is Ellis Island?", 
     "output": "In New York Harbor, known for the Immigration Museum."},
    {"input": "Where is Brooklyn Bridge Park?", 
     "output": "Along the East River in Brooklyn, with views of Manhattan."},
    {"input": "Where is DUMBO?", 
     "output": "Down Under the Manhattan Bridge Overpass, Brooklyn."},
    {"input": "Where is Williamsburg?", 
     "output": "In northern Brooklyn, known for arts and nightlife."},
    {"input": "Where is Coney Island?", 
     "output": "In southern Brooklyn, famous for amusement parks."},
    {"input": "Where is Brighton Beach?", 
     "output": "Next to Coney Island in Brooklyn, known for Russian community."},
    {"input": "Where is Flushing Meadows–Corona Park?", 
     "output": "In Queens, site of the US Open tennis tournament."},
    {"input": "Where is Citi Field?", 
     "output": "In Queens, home of the New York Mets."},
    {"input": "Where is Yankee Stadium?", 
     "output": "In the Bronx, home of the New York Yankees."},
    {"input": "Where is the Bronx Zoo?", 
     "output": "In the Bronx, one of the largest metropolitan zoos in the US."},
    {"input": "Where is the New York Botanical Garden?", 
     "output": "In the Bronx, near the Bronx Zoo."},
    {"input": "Where is Arthur Avenue?", 
     "output": "In the Bronx, known as the 'real' Little Italy."},
    {"input": "Where is Fordham University?", 
     "output": "In the Bronx."},
    {"input": "Where is Pelham Bay Park?", 
     "output": "In the Bronx, the largest park in NYC."},
    {"input": "Where is Staten Island Greenbelt?", 
     "output": "A network of parks and trails in Staten Island."},
    {"input": "Where is Staten Island Zoo?", 
     "output": "In Staten Island."},
    {"input": "Where is Staten Island Ferry Terminal?", 
     "output": "Whitehall Terminal, southern Manhattan."},
    {"input": "Where is Snug Harbor Cultural Center?", 
     "output": "In Staten Island, former sailors’ home and cultural site."},
    {"input": "Where is the Staten Island Greenbelt Nature Center?", 
     "output": "In Staten Island, part of the Greenbelt park system."},
    {"input": "Where is Staten Island Mall?", 
     "output": "In Staten Island, commercial shopping area."},
]
from tinker import types

tokenizer = training_client.get_tokenizer()
def process_example(example: dict, tokenizer) -> types.Datum:
    # Format input/output
    prompt = f"Question: {example['input']}\nAnswer:"
    
    # Encode prompt tokens
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_weights = [0] * len(prompt_tokens)  # don't learn from prompt
    
    # Encode completion tokens
    completion_tokens = tokenizer.encode(f" {example['output']}\n\n", add_special_tokens=False)
    completion_weights = [1] * len(completion_tokens)  # learn from completion
    
    # Combine prompt and completion
    tokens = prompt_tokens + completion_tokens
    weights = prompt_weights + completion_weights
    
    # Shift tokens for next-token prediction
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = weights[1:]

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens)
    )


processed_examples = [process_example(ex, tokenizer) for ex in examples]
 
def batch_generator(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

# Training parameters
num_epochs = 5
batch_size = 16
learning_rate = 1e-4

# Training loop with mini-batches
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    for batch_idx, batch in enumerate(batch_generator(processed_examples, batch_size)):
        fwdbwd_future = training_client.forward_backward(batch, "cross_entropy")
        optim_future = training_client.optim_step(types.AdamParams(learning_rate=learning_rate))

        fwdbwd_result = fwdbwd_future.result()
        optim_result = optim_future.result()

        logprobs = np.concatenate([output['logprobs'].tolist() for output in fwdbwd_result.loss_fn_outputs])
        weights = np.concatenate([ex.loss_fn_inputs['weights'].tolist() for ex in batch])
        avg_loss = -np.dot(logprobs, weights) / weights.sum()
        print(f"Batch {batch_idx+1}/{len(processed_examples)//batch_size + 1} Loss per token: {avg_loss:.4f}")

        sampling_client = training_client.save_weights_and_get_sampling_client(name='nyc-lora-model')

prompt_text = "Question: Where can I see the Statue of Liberty?\nAnswer:"
prompt = types.ModelInput.from_ints(tokenizer.encode(prompt_text))

params = types.SamplingParams(
    max_tokens=20,
    temperature=0.7,   # more creative than greedy
    stop=["\n"]
)

future = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=3)
result = future.result()

print("\nSampled Responses:")
for i, seq in enumerate(result.sequences):
    print(f"{i}: {repr(tokenizer.decode(seq.tokens))}")