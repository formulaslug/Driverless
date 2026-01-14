## Step 1: Start + Enable SSH
Run these in Jetson terminal:

`sudo systemctl enable ssh`

`sudo systemctl start ssh`

## Step 2: Check Status
`systemctl status ssh`

## Step 3: Check IP Adress
Run:
`hostname -I`

Use first IP adress given from command

## Step 4: SSH from your Laptop
Run the following from your laptop:

`ssh formulaslug@{jetson-ip}`

Replace {jetson-ip} with the IP you got from step 3

**It'll ask if you want to coninute say yes !**

## Step 5: Exit Command
type:
`exit` or ctrl+d