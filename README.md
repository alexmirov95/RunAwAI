# Project RunAwAI

#### ECS 170 Spring 2018

#### By: Alex Mirov, Simon Wu, Kyle Chickering, Xin Jin

# RunAwAI
## Abstract
This project explores the integration of a neural network and an evolutionary
genetic algorithm with the Star Craft II API. The goal of this project was to
successfully control a StarCraft II agent via a neural network, and maximize
its fitness through a genetic evolutionary algorithm. The agent uses the PySC2
API to connect with StarCraft II, and a neural network to control its
actions. The fitness function is defined as the survival duration for an agent
in a sandbox environment with hostile units, where surviving for longer is
better. The action space of the agent was limited to movement in order to
achieve a realistic scope for the project. The fitness of an individual is
evaluated by running multiple simulation episodes and averaging the
performance (survival time) for each episode. The neural network uses an
explicit topologically unrestricted implementation. This implementation
provides high degrees of freedom while minimizing the network size. The genetic
algorithm uses tournament selection to select the most fit neural network
(individuals) to breed. Each successive generation of neural networks is
slightly better at surviving in the hostile sandbox environment. After 50
iterative generations, our resulting neural network had increased the average
survival time of an agent by 288%.


## Introduction
After AlphaGo defeated Ke Jie [], the artifical intelligence community has been
looking for the next problem domain to apply learning techniques to. A
potentially surprising candidate for artificial intelligence testing has emerged
in the form of real time strategy games. Real time strategy games work by having
a player make critical descisions in real time against an enemy player. The game
StarCraftII, released by Blizzard entertainment in 2010, is a real time strategy
game that has captured the interest of the artificial intelligence community.
The game presents several challenges for artificial intelligence, the biggest of
which is the magnitude of possible game states.

We present a method of teaching an agent to avoid enemy players for as long as
possibly by using genetic algorithms to evolve neural networks. We interface
our neural network agents using the PySC2 library [] and evolve an agent that
shows a measurable increase in its ability to run away.

## Background
This project combines two proven artificial intelligence paradigms, neural
networks and gentetic algorithms. These methods are traditionally orthogonal,
but there has been a growing body of work in recent years with regards to
evolving neural networks using genetic algorithms. Most notably [ref] and [ref].

### Neural Networks
Neural networks are a method of computing any computable function by using a
network of nodes called __neurons__ connected by a series of edges. This
representation is inspired by the human brain, and has been proven over and over
as a heavy hitter in the world of machine learning. These networks are typically
differentiable, and by calculating error, they can be trained using a method
called backpropogation that changes the strength of the connections between
neurons.

There are two main ways to represent neural networks in software, implicitly and
explicitly. Implicit neural networks use linear algebra and a consistent layered
structure to propogate inputs forward (called feeding forward). With numerical
linear algebra libraries avaliable for almost every major programming language,
this method of representing neural networks allows for incredibly fast processing
of neural networks. However, this representation's greatest strength, its speed,
is the source of its greatest weakness. Implicilty defined neural networks suffer
from the inflexibility of their layered structure. In the human brain, neurons
are not limited to any specific structure, and are free to make connections
that could not be possible in a layered model.

This leads us to explicitly defined neural networks. These networks allow the
specification of new nodes that are disconnected from the traditional layered
model. While this can be advantagous, it can also present problems for the
implementer. Because these models are represented by graphs, we can no longer
use linear algebra libraries to feed forward through the network. This causes
an increase (often significant) in running time. For small networks this is
not an issue, but for larger network this becomes (sometimes prohibitivly)
problematic. 

### Genetic Algorithms
In a similar line of though that led researcher to build neural networks by
looking to nature for inspiration, genetic algorithms exploit natural selection
to create optimal solutions to problems. Research into genetic algorithms
started when researchers realized that there is nothing inherint in evolution
that limits the process to nature []. In fact, by thinking of evolution as
an algorithm in and of itself, we can extend the principle of evolution to
digital systems. This is commonly done by defining what an "individual" and
"fitness function" mean in the digital evolution landscape. The algorithm
proceeds as evolution does in nature by picking the most well adapted
individuals to "breed", and evaluating their "offspring" on the same problem.
This technique has yielded several fascinating results [][][], and often the
algorithm generates novel solutions to difficult problems [][].

## Methodology
Our approach involves combining genetic algorithms and neural networks to
create an agent that learns through evolution how to evade antagonistic
agents. We chose this method as the problem space closely resembles nature's
predator/prey motif. By modeling our agent as an evolving neural network we can
get results that mimic generational evolution in nature. 

## Results

## Conclusion

## Refrences
Kenneth DeJong
Stanly NEAT

