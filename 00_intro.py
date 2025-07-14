# %%

# 1. 

import torch
import torch.nn as nn

# Our toy vocab:

vocab = { 'cat': 0, 'dog': 1, 'fish': 2 }
print(vocab)

# %%

# 2. Embedding layer:

embedding = nn.Embedding( num_embeddings = 3, embedding_dim =2)
print(embedding)


# %%


# 3. Words -> Indices

word_indices = torch.tensor(
    [vocab['cat'], vocab['fish']]
)

vectors = embedding( word_indices ) # Initialized at random
print( vectors )


# %%

# 4. Train the embedding

def get_embed(token):
    y = embedding(torch.tensor([vocab[token]]))
    return y

cat = get_embed('cat')
dog = get_embed('dog')
fish = get_embed('fish')

# Compute distances
cos = nn.CosineSimilarity(dim=1) # this is just a naughty correlation
sim_cat_dog = cos(cat, dog)
sim_cat_fish = cos(cat, fish)

loss = -sim_cat_dog + sim_cat_fish

print('Loss: ', loss.item() )


# %%

# 5. Optimize

print( 'cat v dog, initial: ', cos(cat, dog).item() )        
print( 'cat v fish, initial: ', cos(cat, fish).item() )        

print('\n Initial embedding weights: ')
print( embedding.weight )

optimizer = torch.optim.SGD( embedding.parameters(), 
                            lr=0.1)

for epoch in range(50): # 50-step stoch grad descent
    optimizer.zero_grad()
    #
    cat = get_embed('cat')
    dog = get_embed('dog')
    fish = get_embed('fish')
    #
    sim_cat_dog = cos(cat, dog)
    sim_cat_fish = cos(cat, fish)
    #
    loss = -sim_cat_dog + sim_cat_fish

    # Backprop:
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: , Loss:  {loss.item()} ")


print( 'cat v dog, final: ', cos(cat, dog).item() )        
print( 'cat v fish, final: ', cos(cat, fish).item() )        

print('\n Final embedding weights: ')
print( embedding.weight )

