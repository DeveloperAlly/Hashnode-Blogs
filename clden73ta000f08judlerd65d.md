# 🌈🦄  Build your own AI-generated Art NFT DApp!

> 🦄 **Quick links:**
> 
> * [Video version](https://www.youtube.com/watch?v=nu55bKXnjlU)
>     
> * [GitHub code](https://github.com/DeveloperAlly/bacalhau-fvm-nft)
>     
> * [Bacalhau Docs](https://docs.bacalhau.org/examples/model-inference/stable-diffusion-gpu/)
>     

## 👩‍💻 What we'll do...

This blog will walk you through how to

1. Build an opensource python-based text-to-image script based on Tensorflow (you can also just use the Bacalhau HTTP endpoint if this isn't of interest to you)
    
2. Run this script on Bacalhau (an open p2p off-chain compute platform)
    
3. Create an NFT Contract in Solidity (based on an Open Zeppelin ERC721 contract)
    
4. Deploy the NFT Contract to the Filecoin Virtual Machine (FVM) Hyperspace Testnet with Hardhat
    
5. Front-end interactions - How to interact with Bacalhau text-to-image script and your NFT contract in React
    
6. How to save your NFT Metadata to NFT.Storage
    
7. How to deploy your Front-End DApp to Fleek
    

I've deliberately chosen to use as much open-source and decentralised tech as is available in this stack.

This blog is going to be pretty lengthy (hey - I want to give ALL THE INFO and make sure we're being beginner-friendly and inclusive!) - so feel free to skip through to the parts that are useful to you in the table of contents &lt;3

## 🎮 Try it!

This app is hosted live here

\[Fleek link coming soon\]

## 🏛 Architecture Diagram (kinda)

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1674859307004/bc763d43-90f7-4720-a118-d5b4ccbcadb0.png align="center")

## 🥞 DApp Technology Stack

(get it - it's a pancake stack #sorrynotsorry)

Open Source & Web3-valued from the ground up :)

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1674859217007/1255a637-b266-4245-afaa-a623d5672a2a.png align="center")

* **Smart Contract** \[Solidity, Open Zeppelin\]
    
    * [Solidity](https://docs.soliditylang.org/en/v0.8.17/) is an OO smart contract programming language for Ethereum (EVM) -compatible blockchains
        
    * [Open Zeppelin](https://docs.openzeppelin.com/contracts/4.x/) offers a security-audited implementation library of common smart contract components and contracts
        
* **Smart Contract IDE** \[Hardhat\]
    
    * [Hardhat](https://hardhat.org/docs) is a development environment for editing, compiling, debugging & deploying Ethereum software
        
* **Blockchain Testnet** \[Filecoin Virtual Machine Hyperspace\]
    
    * [FVM Hyperspace](https://fvm.filecoin.io/) is an EVM-compatible testnet built on the Filecoin blockchain
        
* **NFT Metadata Storage** \[NFT.Storage\]
    
    * [NFT.Storage](https://nft.storage/) is a public good built on top of IPFS & Filecoin to store NFT Metadata immutably and persistently & offers free decentralised storage for NFTs and a javascript sdk.
        
* **Front-End** \[NextJS / React + NPM\]
    
    * We probably all know these... right? :P
        
* **Smart Contract Interactions** from client \[Metamask, Ethers, Chainstack RPC Node\]
    
    * Using a [public RPC node](https://www.alchemy.com/overviews/rpc-node) - I can get read-only interactions with my blockchain contract.
        
    * With a [Metamask](https://metamask.io/) provider (or similar wallet that [injects the Ethereum API](https://docs.metamask.io/guide/ethereum-provider.html#table-of-contents) specified by [EIP-1193](https://eips.ethereum.org/EIPS/eip-1193) into the browser), we enable write calls to the blockchain contract.
        
    * [Ethers](https://docs.ethers.org/v5/)js is a library for interacting with EVM-compatible smart contracts
        
* AI **Text-To-Image Stable Diffusion Script** \[Python, Tensorflow\]
    
    * [TensorFlow](https://www.tensorflow.org/) is an open-source machine learning platform and library that provides pre-trained models and other data and ML tools.
        
* Decentralised **Off-Chain Compute** for AI Text-To-Image Generation \[Bacalhau\]
    
    * [Bacalhau](https://docs.bacalhau.org/) is a peer-to-peer open computation network that provides a platform for public, transparent and optionally verifiable computation processes. It's a decentralised off-chain data computation layer.
        
* Decentralised **DApp Deployment** \[Fleek\]
    
    * [Fleek](https://fleek.co/) offers deployment of websites on IPFS & Filecoin. Its the web3 version of Vercel or Netlify - can't say we really have a decentralised app and then deploy it to web2! :D
        

## 🏗️ Building the Python Text-To-Image Script

> 💡 **TLDR Tip** 💡
> 
> This script is already available for use through Bacalhau via the CLI and an HTTP endpoint, so feel free to skip this part.

**Quick Intro to Stable Diffusion**

Stable Diffusion is currently the leading Machine Learning Model for doing text-to-image processing (& is the same model Dall-E uses). It is a type of Deep Learning - a subset of Machine Learning that teaches itself to perform a specific task - in this case converting a text input to an image output.

In this example, we're using a diffusion probabilistic model that uses a transformer to generate images from text.

![](https://lh5.googleusercontent.com/HsqyhPMmmduSXSPuOeNUw1qjXGEgx_YovmKcvF2_oFbVeDjrkx69RhgwkscTmiC3x9PoUSTYf8uAATESivaQ-a21qU2Ka1H9kAwxXXFOvd-JE0Y_Tsg5wfoeKmRBUPSF7RCKKCMahTZeqUfoyRIsUJih3w=s2048 align="center")

Don't worry though - we don't need to go and train a machine learning model for this (though hey - if that's your thing - you totally could!)

Instead, we're going to use a pre-trained model from Google's TensorFlow open-source Machine Learning library in our python script because the ML weights have been pre-calculated for us.

More correctly, we're using an optimised [Keras/TensorFlow implementation fork](https://github.com/fchollet/stable-diffusion-tensorflow/blob/master/text2image.py) of the original ML model.

**The Python Script**

> 🦄 You can find a **complete** walkthrough of how to build and Dockerise this text-to-image script and run it on Bacalhau in both the [Bacalhau docs](https://docs.bacalhau.org/examples/model-inference/stable-diffusion-gpu/) and in this [@BacalhauProject YouTube video](https://www.youtube.com/watch?v=53uY48e1lis).  
> 🦄 You can also run it in this [Google Collabs Notebook](https://colab.research.google.com/github/bacalhau-project/examples/blob/main/model-inference/stable-diffusion-gpu/index.ipynb)

Here's the full python script!

```python
import argparse
from stable_diffusion_tf.stable_diffusion import Text2Image
from PIL import Image
import os

parser = argparse.ArgumentParser(description="Stable Diffusion")
parser.add_argument("--h",dest="height", type=int,help="height of the image",default=512)
parser.add_argument("--w",dest="width", type=int,help="width of the image",default=512)
parser.add_argument("--p",dest="prompt", type=str,help="Description of the image you want to generate",default="cat")
parser.add_argument("--n",dest="numSteps", type=int,help="Number of Steps",default=50)
parser.add_argument("--u",dest="unconditionalGuidanceScale", type=float,help="Number of Steps",default=7.5)
parser.add_argument("--t",dest="temperature", type=int,help="Number of Steps",default=1)
parser.add_argument("--b",dest="batchSize", type=int,help="Number of Images",default=1)
parser.add_argument("--o",dest="output", type=str,help="Output Folder where to store the Image",default="./")

args=parser.parse_args()
height=args.height
width=args.width
prompt=args.prompt
numSteps=args.numSteps
unconditionalGuidanceScale=args.unconditionalGuidanceScale
temperature=args.temperature
batchSize=args.batchSize
output=args.output

generator = Text2Image(
    img_height=height,
    img_width=width,
    jit_compile=False,  # You can try True as well (different performance profile)
)

img = generator.generate(
    prompt,
    num_steps=numSteps,
    unconditional_guidance_scale=unconditionalGuidanceScale,
    temperature=temperature,
    batch_size=batchSize,
)
for i in range(0,batchSize):
  pil_img = Image.fromarray(img[i])
  image = pil_img.save(f"{output}/image{i}.png")
```

The script above simply takes in a text prompt input argument and some other optional parameters and then calls the forked TensorFlow library to generate the image(s) and save them to an outputs file.

All of the heavy lifting done here happens in the section below - this is where the Machine Learning Model does its magic. 🪄

```python
generator = Text2Image(
    img_height=height,
    img_width=width,
    jit_compile=False,
)

img = generator.generate(
    prompt,
    num_steps=numSteps,
    unconditional_guidance_scale=unconditionalGuidanceScale,
    temperature=temperature,
    batch_size=batchSize,
)
```

Great, we can generate images from a text prompt, but um... where to run this GPU-required script..... 🤔🤔

If there's one thing that blockchain technology does not do inherently well, it is large data processing. This is due to the cost of computing over a distributed system to provide other powerful properties like trustlessness and censorship resistance.

Using your local machine for small examples is possible - in fact I did manage to get this particular example working on my (very unhappy about it) Mac M1, however, it was a very long wait on results (game of table tennis anyone?) so, once you start doing bigger data processing, you are going to need more gas (pun intended) and if you don’t have a dedicated server lying around the house, then you’re going to need to use a virtual machine on a cloud computing platform. Not only is that centralised, it's also inefficient - due to the data being an unknown distance from the computation machine, and it can get costly fast. I failed to find any free-tier cloud computing service that offered GPU processing for this (did someone say crypto mining bans..?) and plans came in at **&gt; US$400** a month (no thankyou).

![](https://lh4.googleusercontent.com/LnnSpxEN-MXg4fm2ZBxxowXm2E296RM9Vnth-0Y1lYUZFDRt5RPSBEdwSpRmzQbykG6_BCVHxWcbiEXSwKRjkHHLn9MhLeenOHvH6MUgEXxHUghzOn7xE6CelL-SK453mRw4OzoD6-HmRNhKU-T0XxwUcA=s2048 align="center")

**Bacalhau!**

Luckily though, these problems are some of the issues Bacalhau is trying to solve. Making data processing and computation open and available to everyone and speeding up the processing times is possible in Bacalhau, firstly - by using batch processing across multiple nodes and secondly by putting the processing nodes where the data lives!

Bacalhau is aiming to help democratise the future of data processing by enabling off-chain computation over data without giving up the decentralisation values inherent to IPFS, Filecoin & Web3 more broadly.

[Bacalhau](https://docs.bacalhau.org/) is a peer-to-peer open computation network that provides a platform for public, transparent and optionally verifiable computation processes where users can run Docker containers or Web Assembly images as tasks against *any* data including data stored in IPFS (& soon Filecoin). It even has support for GPU jobs and not at US$400 or more!

![intro | Bacalhau Docs](https://docs.bacalhau.org/assets/images/bacalhau-high-level-view-4866977e82dcfd7b4ec1872ce327f856.png align="center")

**Running the script on Bacalhau**

To run this script, we can Dockerise it for use on Bacalhau. You can follow the [tutorial here](https://docs.bacalhau.org/examples/model-inference/stable-diffusion-gpu/) if you want to learn to do that.  
We can then run it with the Bacalhau CLI with just one line of code (after [installing Bacalhau](https://docs.bacalhau.org/getting-started/installation) with another one-liner):

```bash
bacalhau docker run --gpu 1 ghcr.io/bacalhau-project/examples/stable-diffusion-gpu:0.0.1 -- python main.py --o ./outputs --p "Rainbow Unicorn"
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1674830662856/3ead2e28-b2a7-4128-a770-bbe0284eed6c.png align="center")

In this example though, I'm going to use an HTTP endpoint that connects me to this dockerised stable diffusion script, which I'll show you in the Integrations section!

I'll note here though, that this is a powerful and flexible way to run data computation processes that is also web3 friendly - we're not just limited to this one small model.

Let's move on to the NFT script though! :)

## ⚒️ Building & Deploying the Solidity NFT Script

**The Smart Contract**

The NFT Smart Contract is based on [Open Zeppelin's implementation](https://docs.openzeppelin.com/contracts/4.x/erc721) of ERC721 but uses the ERC721URIStorage version, which includes the metadata standard extensions (so we can pass in our IPFS-addressed metadata - which we'll save on NFT.Storage, to the contract).

This base contract additionally gives us the general functionality of an NFT contract with functions like mint() and transfer() already implemented for us.

You'll notice I've also added a couple of getter functions to fetch data for my front end as well as an Event that will be emitted on-chain each time a new NFT is minted. This gives the ability to listen to on-chain events from the DApp.

> 💡 [Try it out on remix and see all the available functions by clicking this link!](https://remix.ethereum.org/DeveloperAlly/bacalhau-fvm-nft/blob/main/pages/api/hardhat/contracts/BacalhauFRC721.sol) 💡

`BacalhauFRC721.sol`

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.4;

import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
import "@openzeppelin/contracts/utils/Counters.sol";
import "@hardhat/console.sol"; 

contract BacalhauFRC721 is ERC721URIStorage {
/** @notice Counter keeps track of the token ID number for each unique NFT minted in the NFT collection */
    using Counters for Counters.Counter;
    Counters.Counter private _tokenIds;

/** @notice This struct stores information about each NFT minted */
    struct bacalhauFRC721NFT {
        address owner;
        string tokenURI;
        uint256 tokenId;
    }

 /** @notice Keeping an array for each of the NFT's minted on this contract allows me to get information on them all with a read-only front end call */
    bacalhauFRC721NFT[] public nftCollection;
/** @notice The mapping allows me to find NFT's owned by a particular wallet address. I'm only handling the case where an NFT is minted to an owner in this contract - but you'd need to handle others in a mainnet contract like sending to other wallets */
    mapping(address => bacalhauFRC721NFT[]) public nftCollectionByOwner;

/** @notice This event will be triggered (emitted) each time a new NFT is minted - which I will watch for on my front end in order to load new information that comes in about the collection as it happens */
    event NewBacalhauFRC721NFTMinted(
      address indexed sender,
      uint256 indexed tokenId,
      string tokenURI
    );

/** @notice Creates the NFT Collection Contract with a Name and Symbol */
    constructor() ERC721("Bacalhau NFTs", "BAC") {
      console.log("Hello Fil-ders! Now creating Bacalhau FRC721 NFT contract!");
    }

/** 
@notice The main function which will mint each NFT.
The ipfsURI is a link to the ipfs content identifier hash of the NFT metadata stored on NFT.Storage. This data minimally includes name, description and the image in a JSON.
*/
    function mintBacalhauNFT(address owner, string memory ipfsURI)
        public
        returns (uint256)
    {
        // get the tokenID for this new NFT
        uint256 newItemId = _tokenIds.current();

        // Format info for saving to our array
        bacalhauFRC721NFT memory newNFT = bacalhauFRC721NFT({
            owner: msg.sender,
            tokenURI: ipfsURI,
            tokenId: newItemId
        });

        //mint the NFT to the chain
        _mint(owner, newItemId);
        //Set the NFT Metadata for this NFT
        _setTokenURI(newItemId, ipfsURI);

        _tokenIds.increment();
        
        //Add it to our collection array & owner mapping
        nftCollection.push(newNFT);
        nftCollectionByOwner[owner].push(newNFT);

        // Emit an event on-chain to say we've minted an NFT
        emit NewBacalhauFRC721NFTMinted(
          msg.sender,
          newItemId,
          ipfsURI
        );

        return newItemId;
    }

    /**
     * @notice helper function to display NFTs for frontends
     */
    function getNFTCollection() public view returns (bacalhauFRC721NFT[] memory) {
        return nftCollection;
    }

    /**
     * @notice helper function to fetch NFT's by owner
     */
    function getNFTCollectionByOwner(address owner) public view returns (bacalhauFRC721NFT[] memory){
        return nftCollectionByOwner[owner];
    }
```

**Requirements**

I'll be deploying this contract to the [Filecoin Virtual Machine Hyperspace Testnet](https://github.com/filecoin-project/testnet-hyperspace), but you could deploy this contract to any EVM-compatible chain including Polygon, BSC, Optimism, Arbitrum, Avalanche and more. You could even tweak your front end to make a multi-chain NFT (hint: [this repo](https://github.com/DeveloperAlly/filecoin-expanded-nft-starter))!

To deploy to Hyperspace Testnet we'll need to

1. [Set up & connect](https://docs.filecoin.io/developers/smart-contracts/how-tos/add-to-metamask/) Metamask Wallet to Hyperspace Testnet
    
2. Get some test tFIL funds from a faucet ([Yoga](https://hyperspace.yoga/#faucet) or [Zondax](https://beryx.zondax.ch/faucet))
    

**Deploying the Smart Contract with Hardhat**

I'm using hardhat to deploy this contract to the Hyperspace testnet.

> 🛸 **Hyperspace RPC & BlockExplorer Options:**
> 
> | Public RPC Endpoints | BlockExplorer's |
> | --- | --- |
> | [https://filecoin-hyperspace.chainstacklabs.com/rpc/v0](https://filecoin-hyperspace.chainstacklabs.com/rpc/v0) | [https://beryx.zondax.ch/](https://beryx.zondax.ch) |
> | [https://hyperspace.filfox.info/rpc/v0](https://hyperspace.filfox.info/rpc/v0) | [https://fvm.starboard.ventures/contracts/](https://fvm.starboard.ventures/contracts/) |
> | [https://rpc.ankr.com/filecoin\_testnet](https://rpc.ankr.com/filecoin_testnet) | [https://explorer.glif.io/?network=hyperspacenet](https://explorer.glif.io/?network=hyperspacenet) |
> | **Open API**: https://beryx.zondax.ch/ | [https://hyperspace.filfox.info/en](https://hyperspace.filfox.info/en) |

For the config set-up, we can choose from any of the available public RPC endpoints.

`hardhat.config.ts`

```typescript
import '@nomicfoundation/hardhat-toolbox';
import { config as dotenvConfig } from 'dotenv';
import { HardhatUserConfig } from 'hardhat/config';
import { resolve } from 'path';

//Import our customised tasks
// import './pages/api/hardhat/tasks';

const dotenvConfigPath: string = process.env.DOTENV_CONFIG_PATH || './.env';
dotenvConfig({ path: resolve(__dirname, dotenvConfigPath) });

// Ensure that we have all the environment variables we need.
const walletPrivateKey: string | undefined = process.env.WALLET_PRIVATE_KEY;
if (!walletPrivateKey) {
  throw new Error('Please set your Wallet private key in a .env file');
}

const config: HardhatUserConfig = {
  solidity: '0.8.17',
  defaultNetwork: 'filecoinHyperspace',
  networks: {
    hardhat: {},
    filecoinHyperspace: {
      url: 'https://api.hyperspace.node.glif.io/rpc/v1',
      chainId: 3141,
      accounts: [process.env.WALLET_PRIVATE_KEY ?? 'undefined'],
    },
    // bleeding edge often-reset FVM testnet
    filecoinWallaby: {
      url: 'https://wallaby.node.glif.io/rpc/v0',
      chainId: 31415,
      accounts: [process.env.WALLET_PRIVATE_KEY ?? 'undefined'],
      //explorer: https://wallaby.filscan.io/ and starboard
    },
  },
// I am using the path mapping so I can keep my hardhat deployment within the /pages folder of my DApp and therefore access the contract ABI for use on my frontend
  paths: {
    root: './pages/api/hardhat',
    tests: './pages/api/hardhat/tests', //who names a directory in the singular?!!! Grammarly would not be happy
    cache: './pages/api/hardhat/cache',
  },
};

export default config;
```

And to deploy the smart contract we create a deploy script - note that I'm specifically setting the Wallet address here as the signer (owner) - there are a few mapping errors still being worked in FEVM out at the time of writing that can cause some odd behaviour.

`deploy/deployBacalhauFRC721.ts`

```typescript
import hre from 'hardhat';

import type { BacalhauFRC721 } from '../typechain-types/contracts/BacalhauFRC721';
import type { BacalhauFRC721__factory } from '../typechain-types/factories/contracts/BacalhauFRC721__factory';

async function main() {
  console.log('Bacalhau721 deploying....');

// !!!needed as hardhat's default does not map correctly to the FEVM
  const owner = new hre.ethers.Wallet(
    process.env.WALLET_PRIVATE_KEY || 'undefined',
    hre.ethers.provider
  );
  const bacalhauFRC721Factory: BacalhauFRC721__factory = <
    BacalhauFRC721__factory
  > await hre.ethers.getContractFactory('BacalhauFRC721', owner);

  const bacalhauFRC721: BacalhauFRC721 = <BacalhauFRC721>(
    await bacalhauFRC721Factory.deploy()
  );
  await bacalhauFRC721.deployed();
  console.log('bacalhauFRC721 deployed to ', bacalhauFRC721.address);
  // optionally log to a file here
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
```

To deploy, run the above script in the terminal by using the following code (NB: since we've set the default network to filecoinHyperspace in our config, it's not necessary to pass a flag for the network though this is shown below)

`> cd ./pages/hardhat/deploy/`

```bash
npx hardhat run ./deployBacalhauFRC721.ts --network filecoinHyperspace
```

Celebrate! We've just deployed our NFT contract to the Filecoin hyperspace testnet!

![Dancing Unicorn GIFs | Tenor](https://media.tenor.com/MvvXjGVrnMQAAAAC/dancing-unicorn-unicorn.gif align="center")

## 🎬 Building the Front-End Interactions

Wooo onto the pretty part... and also the glue that holds it all together here :)

To build the front end, I'm using NextJS and Typescript. Though, to be honest - I'm not taking advantage of any of NextJS's SSR (server-side rendering) features and I don't even use their page routing (since it's a single-page Dapp), so you could really just go with a vanilla React set up (or any framework of your choice of course!).

As for the typescript... well, I built this in a bit of a rush and have to admit this is not a very good example of Typescript - the vars seem happy though... ;)

![](https://i.imgflip.com/792lnu.jpg align="center")

Anyhoo - the main point of this section is not to show you how to code a front end, but to show you how to interact with the smart contract, Bacalhau (with our stable diffusion ML model) and of course, NFT.Storage - #NotOnIPFSNotYourNFT.

### Complete Flow

\[todo: build a flow chart diagram\]

* The user enters a text prompt into the input field -&gt;
    
* Clicks Generate Images Button -&gt; Calls Bacalhau Job to Generate images
    
* Bacalhau Job completes -&gt; formats return into NFT Metadata JSON object
    
* User Clicks Mint NFT Button -&gt; NFT.Storage is called to save the NFT Metadata and returns with an IPFS CID for the folder -&gt; The mint NFT function of the smart contract is called with this IPFS\_URI to mint an NFT with this metadata -&gt;
    
* !! \[FEVM gotcha\] -&gt; here we would generally wait for the TX (transaction hash) of this result to return, but it's currently not working, so instead we're using a contract event listener to find out when this completes.
    
* Done! -&gt; Can now re-fetch any display data and give the user status success feedback on the minting.
    

Nice - let's see how we implement this in code!

### **Bacalhau Interactions**

Creating the front-end API endpoint for Bacalhau is documented in [this project report](https://bacalhau.substack.com/p/bacalhau-project-report-jan-25-2022) by engineer [Luke Marsden](https://twitter.com/lmarsden).

The API currently *only* directly hits the stable diffusion scripts documented in this blog, however, the team is in the process of extending it into a more generic API so that you can call any of the examples, and also your own deployed scripts from an HTTP REST API. Keep an eye on this [here](https://github.com/filecoin-project/bacalhau) or in the #bacalhau channel in [FilecoinProject slack.](https://filecoinproject.slack.com/)

`>run/test in terminal`

```bash
curl -XPOST -d '{"prompt": "rainbow unicorn"}' 'http://dashboard.bacalhau.org:1000/api/v1/stablediffusion';
```

`>react / typescript code`

```typescript
import { CID } from 'multiformats/cid';

export const callBacalhauJob = async (promptInput: string) => {
  //Bacalahau HTTP Stable Diffusion Endpoint
  const url = 'http://dashboard.bacalhau.org:1000/api/v1/stablediffusion';
  const headers = {
    'Content-Type': 'application/x-www-form-urlencoded',
  };
  const data = {
    prompt: promptInput, //The user text prompt!
  };
  /* FETCH FROM BACALHAU ENDPOINT */
  const cid = await fetch(url, {
    method: 'POST',
    body: JSON.stringify(data),
    headers: headers,
  })
    .then(async (res) => {
      let body = await res.json();
      if (body.cid) {
/* Bacalhau returns a V0 CID which we want to convert to a V1 CID for easier usage with http gateways (ie. displaying the image on web), so I'm using the IPFS multiformats package to convert it here */
        return CID.parse(body.cid).toV1().toString();
      }
    })
    .catch((err) => {
      console.log('error in bac job', err);
    });
  return cid;
};
```

This function will return an IPFS CID (content identifier) with a folder structure like the one below. The image can then be found under `/outputs/image0.png`.

> 💡 [**See it for yourself by clicking here**](https://bafybeicwzflyf5sole4itagpguf2oktscx3gtgoappmj3nqqqdhfrwzhle.ipfs.nftstorage.link/)**!** 💡

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1674856969385/6b8d9f88-ac55-4565-bcf6-ac2173cb6dcd.png align="center")

![](https://bafybeicwzflyf5sole4itagpguf2oktscx3gtgoappmj3nqqqdhfrwzhle.ipfs.nftstorage.link/outputs/image0.png align="center")

Ahhh rainbow unicorns... what's not to like!

### **NFT.Storage**

NFT.Storage is a public good (aka free) that makes it easy to store NFT Metadata perpetually on IPFS & Filecoin with either a javascript or HTTP SDK.

NFT Metadata is a JSON document that looks something like the example below -which is taken directly from the Open Zeppelin docs:

![](https://lh4.googleusercontent.com/hvXnX_VZ8d9Lr3TqHZWIZatMh_mfsFI9ZNCpJgjMeFu54UNBrfw3sePIMDDZ8EfWFrzaeLzSCd0Zn_zRmr5U-sC6_c6Nn5_2vYMwbtx68L0fFfOLk6fEG8coXNgP3DTra6pr7AJbm2cXejG_4NsXWI3EZg=s2048 align="left")

When creating NFTs, it's important to note that unless you are storing the metadata on-chain (which can become prohibitively expensive for large files), then in order to conform to the 'non-fungibility' of a token, you need storage that is persistent, reliable and ***immutable.***

If your NFT has a location-based address like the above example, then it's fairly simple for this location path to be switched out after a sale, meaning the NFT you thought you bought becomes something entirely different - or a literal rug pull in the case below where the NFT creator switched out the art images for pictures of rugs.

![](https://lh4.googleusercontent.com/Oqij2aelePDTaE0yoDBnIx77-IZvCCQYyYNPFkvbUI12Hp6_KQ2KiOaTuCa3ccfR76vnLAOYzj3j9K9W7Y_-u_SbpJwWC5kVVLHl_qleTRrmC9FgCWFm1ch_iWx66clSzjRxfe3zuyZUepU0F3ALnFIr=s2048 align="left")

![](https://lh6.googleusercontent.com/azZfRPoeJbjJLvntGgSZZFJLgqPpIYK3unC_Pym0bStsRr0KfRzaCxbXxpM5VV1kapxr47pigYC5qUfO2n9FsijFrtmO1xTbBgD1h1s4lyq5punKH1NRdy6VGPOi58VcjPr1QA-qVyGsk5FC_65sIkYk=s2048 align="left")

Something even Open Zeppelin warns about!

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1674907241854/879ff3ff-7e6f-462a-97f3-0456c1ae80cd.png align="center")

Using NFT.Storage means that we get an immutable IPFS file CID (**content** - not location - **id**entifier) for our metadata which is not just pinned to IPFS but also then stored to Filecoin for persistence.  
You'll just need to sign up for NFT.Storage and get an [API key](https://nft.storage/manage/) (to save in your .env file) for this one.

`.env example`

```typescript
NEXT_PUBLIC_NFT_STORAGE_API_KEY=xxx
```

We also need to make sure that we have created a properly formed Metadata JSON - because while FVM doesn't (yet!) have NFT Marketplaces... we do want to make sure that when it's adopted our NFT still hopefully conforms to the standard.

```typescript
import { NFTStorage } from 'nft.storage';

//connect to NFT.Storage Client
const NFTStorageClient = new NFTStorage({
   token: process.env.NEXT_PUBLIC_NFT_STORAGE_API_KEY,
});
 
const createNFTMetadata = async (
    promptInput: string,
    imageIPFSOrigin: string, //the ipfs path eg. ipfs://[CID]
    imageHTTPURL: string //an ipfs address fetchable through http for the front end to use (ie. including an ipfs http gateway on it like https://[CID].ipfs.nftstorage.link)
  ) => {
    console.log('Creating NFT Metadata...');
    let nftJSON;
 // let's get the image data Blob from the IPFS CID that was returned from Bacalhau earlier...
    await getImageBlob(status, setStatus, imageHTTPURL).then(
      async (imageData) => {
// Now let's create a unique CID for that image data - since we don't really want the rest of the data returned from the Bacalhau job..
        await NFTStorageClient.storeBlob(imageData)
          .then((imageIPFS) => {
            console.log(imageIPFS);
//Here's the JSON construction - only name, description and image are required fields- but I also want to save some other properties like the ipfs link and perhaps you have other properties that give your NFT's rarity to add as well
            nftJSON = {
              name: 'Bacalhau Hyperspace NFTs 2023',
              description: promptInput,
              image: imageIPFSOrigin, 
              properties: {
                prompt: promptInput,
                type: 'stable-diffusion-image',
                origins: {
                  ipfs: `ipfs://${imageIPFS}`,
                  bacalhauipfs: imageIPFSOrigin,
                },
                innovation: 100,
                content: {
                  'text/markdown': promptInput,
                },
              },
            };
          })
          .catch((err) => console.log('error creating blob cid', err));
      }
    );
    return nftJSON;
  };
```

Now let's store this metadata to NFT.Storage!

```typescript
await NFTStorageClient.store(nftJson)
  .then((metadata) => {
    // DONE! - do something with this returned metadata!
    console.log('NFT Data pinned to IPFS & stored on Filecoin!');
    console.log('Metadata URI: ', metadata.url);
    // once saved we can use it to mint the NFT
    // mintNFT(metadata);
  })
  .catch((err) => {
    console.log('error uploading to nft.storage');
  });
```

Woot - we have our image from Bacalhau, we've saved our metadata immutably and persistently with NFT.Strorage, now let's mint our NFT!

> 💡 **Quick Tip** 💡  
> NFT.Storage also offers a range of other [**API calls**](https://nftstorage.github.io/nft.storage/client/classes/lib.NFTStorage.html) like storeCar & storeDirectory as well as a **status() function** - which returns the IPFS pinning and Filecoin storage deals of a CID -&gt; this could be a pretty cool addition for a FEVM DApp (or NFT implementation on FEVM once FEVM hits mainnet release) for checking on NFTs status.

### **Contract Interactions**

There are 3 types of interactions here (and a few FEVM gotcha's - beta tech is always going to have some quirky <s>bugs</s> features!)

* read-only calls to retrieve data from the chain without mutating it
    
* write calls that require a wallet to sign and pay gas ie. functions that change the state of the chain, like minting the NFT!
    
* event listeners - that listen for events emitted from the contract
    

For all of these functions, we'll use the [ethers.js library](https://docs.ethers.org/v5/) - a lightweight wrapper for the Ethereum API, to connect to our contract and perform calls to it.

Connecting to the contract in **read** mode with a public RPC:

```typescript
//The compiled contract found in pages/api/hardhat/artifacts/contracts  
import BacalhauCompiledContract from '@Contracts/BacalhauFRC721.sol/BacalhauFRC721.json';
//On-chain address of the contract
const contractAddressHyperspace = '0x773d8856dd7F78857490e5Eea65111D8d466A646'; 
//A public RPC Endpoint (see table from contract section)
const rpc = 'https://api.hyperspace.node.glif.io/rpc/v1';

const provider = new ethers.providers.JsonRpcProvider(rpc);
const connectedReadBacalhauContract = new ethers.Contract(
      contractAddressHyperspace,
      BacalhauCompiledContract.abi,
      provider
    );
```

Listening for events on the contract. Since this is a read-only (get) event, we can use the public RPC to listen for event emissions on-chain.

```typescript
//use the read-only connected Bacalhau Contract
connectedReadBacalhauContract.on(
    // Listen for the specific event we made in our contract
    'NewBacalhauFRC721NFTMinted',
    (sender: string, tokenId: number, tokenURI: string) => {
        //DO STUFF WHEN AN EVENT COMES IN
        // eg. re-fetch NFT's, store in state and change page status
    }
);
```

Connecting to the contract in **write** mode - this requires that the Ethereum object is being injected into the web browser by a wallet so that a user can sign for a transaction and pay for gas - which is why we're checking for a window.ethereum object.

```typescript
//Typescript needs to know window is an object with potentially and ethereum value. There might be a better way to do this? Open to tips!
declare let window: any;
//The compiled contract found in pages/api/hardhat/artifacts/contracts  
import BacalhauCompiledContract from '@Contracts/BacalhauFRC721.sol/BacalhauFRC721.json';
//On-chain address of the contract
const contractAddressHyperspace = '0x773d8856dd7F78857490e5Eea65111D8d466A646'; 

//check for the ethereum object
if (!window.ethereum) {
    //ask user to install a wallet or connect
    //abort this
}
// else there's a wallet provider
else {
// same function - different provider - this one has a signer - the user's connected wallet address
   const provider = new ethers.providers.Web3Provider(window.ethereum);
   const contract = new ethers.Contract(
      contractAddressHyperspace,
      BacalhauCompiledContract.abi,
      provider
    );
   const signer = provider.getSigner();
   const connectedWriteBacalhauContract = contract.connect(signer);
}
```

Calling the mint Function using the write connected contract.

First, ensure we have a wallet address from the user and that we're on the FVM Hyperspace chain. Here are a few helpful wallet functions you might want, including how to check the chainId, and how to programmatically add the Hyperspace network to Metamask / wallet.  
You can interact with wallets using the Ethereum object directly or using ethers.js.

```typescript
declare let window: any;

const fetchWalletAccounts = async () => {
  console.log('Fetching wallet accounts...');
  await window.ethereum //use ethers?
    .request({ method: 'eth_requestAccounts' })
    .then((accounts: string[]) => {
      return accounts;
    })
    .catch((error: any) => {
      if (error.code === 4001) {
        // EIP-1193 userRejectedRequest error
        console.log('Please connect to MetaMask.');
      } else {
        console.error(error);
      }
    });
};

const fetchChainId = async () => {
  console.log('Fetching chainId...');
  await window.ethereum
    .request({ method: 'eth_chainId' })
    .then((chainId: string[]) => {
      return chainId;
    })
    .catch((error: any) => {
      if (error.code === 4001) {
        // EIP-1193 userRejectedRequest error
        console.log('Please connect to MetaMask.');
      } else {
        console.error(error);
      }
    });
};

//!! This function checks for a wallet connection WITHOUT being intrusive to to the user or opening their wallet
export const checkForWalletConnection = async () => {
  if (window.ethereum) {
    console.log('Checking for Wallet Connection...');
    await window.ethereum
      .request({ method: 'eth_accounts' })
      .then(async (accounts: String[]) => {
        console.log('Connected to wallet...');
        // Found a user wallet
        return true;
      })
      .catch((err: Error) => {
        console.log('Error fetching wallet', err);
        return false;
      });
  } else {
    //Handle no wallet connection 
    return false;
  }
};

//Subscribe to changes on a user's wallet
export const setWalletListeners = () => {
  console.log('Setting up wallet event listeners...');
  if (window.ethereum) {
    // subscribe to provider events compatible with EIP-1193 standard.
    window.ethereum.on('accountsChanged', (accounts: any) => {
      //logic to check if disconnected accounts[] is empty
      if (accounts.length < 1) {
        //handle the locked wallet case
      }
      if (userWallet.accounts[0] !== accounts[0]) {
        //user has changed address
      }
    });

    // Subscribe to chainId change
    window.ethereum.on('chainChanged', () => {
      // handle changed chain case
    });
  } else { 
        //handle the no wallet case
    }
};

export const changeWalletChain = async (newChainId: string) => {
  console.log('Changing wallet chain...');
  const provider = window.ethereum;
  try {
    await provider.request({
      method: 'wallet_switchEthereumChain',
      params: [{ chainId: newChainId }], //newChainId
    });
  } catch (error: any) {
    alert(error.message);
  }
};

//AddHyperspaceChain
export const addHyperspaceNetwork = async () => {
  console.log('Adding the Hyperspace Network to Wallet...');
  if (window.ethereum) {
    window.ethereum
      .request({
        method: 'wallet_addEthereumChain',
        params: [
          {
            chainId: '0xc45',
            rpcUrls: [
              'https://hyperspace.filfox.info/rpc/v0',
              'https://filecoin-hyperspace.chainstacklabs.com/rpc/v0',
            ],
            chainName: 'Filecoin Hyperspace',
            nativeCurrency: {
              name: 'tFIL',
              symbol: 'tFIL',
              decimals: 18,
            },
            blockExplorerUrls: [
              'https://fvm.starboard.ventures/contracts/',
              'https://hyperspace.filscan.io/',
              'https://beryx.zondax.chfor',
            ],
          },
        ],
      })
      .then((res: XMLHttpRequestResponseType) => {
        console.log('added hyperspace successfully', res);
      })
      .catch((err: ErrorEvent) => {
        console.log('Error adding hyperspace network', err);
      });
  }
};
```

Call the contract mint function in write mode....

```typescript
// Pass in the metadata return from saving to NFT.Storage
const mintNFT = async (metadata: any) => {  
    await connectedWriteBacalhauContract
    // The name of our function in our smart contract
    .mintBacalhauNFT(
      userWallet.accounts[0], //users account to use
      metadata.url //test ipfs address
    )
    .then(async (data: any) => {
      console.log('CALLED CONTRACT MINT FUNCTION', data);
      await data
        .wait()
        .then(async (tx: any) => {
          console.log('tx', tx);
//CURRENTLY NOT RETURNING TX - (I use event triggering to know when this function is complete)
          let tokenId = tx.events[1].args.tokenId.toString();
          console.log('tokenId args', tokenId);
          setStatus({
            ...INITIAL_TRANSACTION_STATE,
            success: successMintingNFTmsg(data),
          });
        })
        .catch((err: any) => {
          console.log('ERROR', err);
          setStatus({
            ...status,
            loading: '',
            error: errorMsg(err.message, 'Error minting NFT'),
          });
        });
    })
    .catch((err: any) => {
      console.log('ERROR1', err);
      setStatus({
        ...status,
        loading: '',
        error: errorMsg(
          err && err.message ? err.message : null,
          'Error minting NFT'
        ),
      });
    });
}
```

Wooo - NFT Minted!! Unicorn dance mode time!

![Dancing Unicorn GIFs | Tenor](https://media.tenor.com/MvvXjGVrnMQAAAAC/dancing-unicorn-unicorn.gif align="center")

## 📺 Deploying the front end to Fleek

\[coming soon\]

## 🌟 Final Thoughts: Possibilities for AI & Blockchain

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1674954231170/942a1402-fa20-4ff1-bbc9-6d76bded634d.jpeg align="center")

Bacalhau lends itself well to performing repetitive, deterministic processing jobs over data.

* ETL Processes
    
* Machine Learning & AI
    
* IOT data integration
    
* Batch Processing including for
    
    * Financial and market data
        
* Video & Image processing - great for creatives
    

There are multiple examples in the [Bacalhau docs](https://docs.bacalhau.org/) of how to achieve some of the above too.

While Bacalhau is busy building out an integration to directly call Bacalhau from FEVM Smart contracts, here's some thoughts on Bacalhau x FVM collaborations:

* Help Onboarding and Offboarding of Filecoin data in the future
    
* Help build a reputation and Quality of Service layer for Filecoin by processing data retrieved on-chain about deals and storage providers.
    
* Bacalhau could provide computation for market & payment data
    
* Bacalhau could help with processing data from DAO’s & DataDAOs
    
* Bacalhau could help empower more automation for creative endeavours like video and images processing
    
* Bacalhau can enable game and metaverse data processing including for VR & AR.
    
* Bacalhau, IOT & Simulations are possible
    
* AI & ML Applications
    

## 🐠 The Bacalhau Roadmap

**We're currently building out a way for you to run Bacalhau directly from your smart contracts!!!!**  
This project is called Project Frog / Project Lilypad - and will be an integration layer that will enable calling Bacalhau jobs from FEVM smart contracts.

Keep an eye on the progress of this by signing up for our newsletter or joining the below socials.

## ✍️ Keep in touch!

Congrats if you read all the way through!!!

I'd appreciate a like, comment, follow or share if this was useful to you! &lt;3

Keep in touch with Bacalhau!

* Twitter [@BacalhauProject](https://twitter.com/BacalhauProject)
    
* YouTube [@BacalhauProject](https://www.youtube.com/@bacalhauproject/playlists)
    
* Filecoin Project Slack #bacalhau [@filecoinproject](https://filecoinproject.slack.com)
    
* Github @[bacalhau.org](http://bacalhau.org)
    
* Forum [github.com/filecoin-project/bacalhau/discussions](https://github.com/filecoin-project/bacalhau/discussions)
    

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1674791065491/9f66f9b5-858f-4db6-8ed7-83ffd0c63c80.png align="center")

With ♥️ [DeveloperAlly](https://twitter.com/DeveloperAlly)