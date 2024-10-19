It all starts with this question:  _why does **“complexity”** or “interestingness” of physical systems seem to increase with time and then hit a maximum and decrease, in contrast to the entropy, which of course increases monotonically?_

answer :

There are two terms here, we need to understand that. 
1. complexity
2. entropy

entropy : It comes from second law of thermodynamics, that heat always from spontaneously from hot to cold region. so, as in it is the physical property of thermodynamic system, it tells if a process is not possible, even if it obeys first law of thermodynamics (law of conservation of energy).

example : acc. to first law, a cup (stable entropy or low entropy) can fall of a table and break(higher entropy) on contact with the floor, as well "jump back" to the table and restore to its original state. but acc. to second law the table cannot go back to its original composition, as systems never go back to lower entropy. It is natural as well, coz cup can never go back to be ok, if that happened it would be cool. It also follows a concept called [arrow of time](https://en.wikipedia.org/wiki/Arrow_of_time) .

and according to the image in blog, we could never "unmix" the milk and coffee after it is in mixed state.

[![](https://149663533.v2.pressablecdn.com/coffee-small.jpg)](https://149663533.v2.pressablecdn.com/coffee-lrg.jpg)

>As Sean points out in his wonderful book [From Eternity to Here](http://www.amazon.com/Eternity-Here-Quest-Ultimate-Theory/dp/0452296544/lecturenotesonge), the Second Law is _almost_ a tautology: how could a system _not_ tend to evolve to more “generic” configurations?  if it didn’t, those configurations wouldn’t _be_ generic!  So the real question is not why the entropy is increasing, but why it was ever low to begin with.  In other words, why did the universe’s initial state at the big bang contain so much order for the universe’s subsequent evolution to destroy?  I won’t address that celebrated mystery in this post, but will simply take the low entropy of the initial state as given.

I don't really agree that the initial state had order, actually it was single point, highly disordered, heated and dense, so much so that it "burst" or big banged. It was expanding and then it got cooler (medium entropy, highly intersting) which is right now i guess. 

but we will leave it to that.

### author's definition of complexity

in the start it is assumed to be low, or close to zero and according to law of thermodynamics it should be less at the end stage, because of nearing towards equilibrium.

but in the intermediate steps, how does it work?

so, author introduced [sophistication](https://peterbloem.nl/files/two-problems-for-sophistication.presentation.pdf) which is based on **Kolmogorov complexity**. 

so, Kolmogorov complexity is about the information quantification. it tells how we quantify the x amount of information.

simply put, _Kolmogorov complexity_ of a string x is the length of the shortest computer program that outputs x.\
 or
 _if it takes 24 bits to define "CAT" then it must be 24 bits at most_.
 >note: assuming 8bits = 1byte, so 3 character word using 8 bits per character is equal to 24 bits > 

and we can also say, the shortest possible description required to define something is the most amount of information that it can contain.

but, Kolmogorov complexity doesn't capture the interestingness of the data, which is between the two extremes (simple and complex) of the data.

here, comes the sophistication, which tries to explain this.

so, the basic idea is not to measure all information in a **STRING** but split up in "RESIDUAL" and **STRUCTURAL** part.

it is done by making  model class, or i call it property class.
then the data is described, by first describing the model and then provide whatever is required to get from 
model -> data. here, it counts the structural info, called as _two-part coding_.

now, coming back to the blog.

to solve the problem, two ways are proposed. 

1. probabilistic systems : - 