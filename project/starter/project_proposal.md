# Project Proposal

#  STEP 1:   Review the Target Student, Prerequisites, and Learning Objectives

| Target Student  *Why are the students here? List their current job titles and the titles they aspire to.* |  |
| :---- | :---- |
| **Prerequisite Skills**  *What should the learner be able to do before they start the project?* |   |
| **Udacity Programs**  *Which Udacity courses or lessons should the learner complete before they start the project?* |  |
| **Project Learning Objectives**  *What will the student be able to do as a result of completing this project? Which skills will they demonstrate in the project?* |   |

#  STEP 2:  Generate a scenario or context

| Project Scenario  *Who is the learner pretending to be in the project and what problem are they solving? Describe the scenario or context for the project. The scenario should be engaging and relevant to real-world practitioners.*  |  |
| :---- | :---- |

# 

# STEP 3:  Describe the project details

| Project Title  *Give your project a snappy and meaningful title to engage the learner* |  |
| :---- | :---- |
| **Project Overview**  *Briefly describe the project, including inputs, a description of the major deliverables, and a high-level description of the required tasks*  |  |
| **Project Length**  *How long should it take an average student to complete the project (in hours)?A typical Udacity project should take 5 hours, but advanced projects can take longer.* |  |
| **Technical Requirements and Dependencies**  *Which programming languages, packages, services, and tools will the student need to build this project?*  *Are there any costs or account requirements associated with these tools or services?*  |       |
| **Project Steps**  *What will the student do to complete the project? Share details about the deliverables and the tools the learner will use.*  ***Note:** these are not detailed project instructions. You'll write those in the next phase when you build the project.* |   |

# 

# 

# ***Example Project Proposal***

#  STEP 1:  Review the Target Student, Prerequisites, and Learning Objectives

| Target Student | *An IT professional/network and system administrator seeking to make a career shift to cybersecurity or add to their experience and skill-set to distinguish themselves within their current professional role. A current tier one Security Analyst or SOC analyst looking to practice skills and grow in their current role. A recent computer science graduate looking to obtain an entry-level role in cybersecurity. Graduate of the Udacity Intro to Cybersecurity Nanodegree looking to expand their skillset* |
| :---- | :---- |
| **Prerequisite Skills** | *Familiar with security fundamentals including core security principles, critical security controls, and best practices for securing information.  Knowledgeable in networking and operating systems. Experience using command line tools (Unix/Linux shell or PowerShell)* |
| **Udacity Programs** | *Udacity Monitoring, Logging, and Responding to Incidents course (cd0291) Foundations of Monitoring and Logging Incident Detection Monitoring and Logging Incident Handling* |
| **Project Learning Objectives** | *Analyze alerts from an Intrusion Detection System (IDS)  Determine whether alerts are incidents (true positives) or false positives. Create rules in an Intrusion Detection System (IDS) based on specified network traffic. Monitor systems for suspicious activity. Monitor network traffic and capture packets in order to investigate suspicious activity.  Use a SIEM to collect and analyze network and host-based data.  Follow incident handling procedures  Document incident evidence, construct a timeline, and detail mitigation next steps.* |

#  STEP 2:  Generate a scenario or context

| Project Scenario  *Who is the learner pretending to be in the project and what problem are they solving? Describe the scenario or context for the project. The scenario should be engaging and relevant to real-world practitioners.*  | *The learner is a junior security analyst at a fictional company, Yoyodyne, who is filling in for a senior analyst who is on vacation. Some suspicious alerts are returned by the company's (IDS) Intrusion Detection System.  The learner must analyze the alerts in the IDS, set up a SIEM to collect and analyze data, monitor systems for suspicious activity, document the evidence, and follow the incident handling procedures to respond to any issues you uncover.* |
| :---- | :---- |

# 

# STEP 3:  Describe the key details of the project

| Project Title | *Intrusion Detection and Response at Yoyodyne* |
| :---- | :---- |
| **Project Overview** | *In this project, students will use skills they have learned to review and make decisions about network-based IDS (Intrusion Detection System) alerts. They will analyze network packet details using Wireshark. They will configure and use a SIEM (Security Information and Event Management) to analyze additional Network Security Monitoring (NSM) data to identify both security incidents and false positives. The project will include using Sguil to review IDS alerts (provided to the students). Using features of Sguil, they will examine alert details and determine whether the alerts are false positives. In the case of true positives, students will start initial incident documentation, including recommended next steps. The project will also include capturing network packets matching specified criteria using tcpdump. Students will analyze the packets using Wireshark (or tshark) and examine application-level data. Using the data found, students will develop a new Snort/Suricata rule to detect matching packets. Additionally, students will configure centralized logging. Using Splunk, one of the most widely used SIEM platforms, students will install and configure a log management system. Students will then identify additional events of interest and create dashboards and reports to help notify analysts of important network- and host-based events.* |
| **Project Length** | *5 hours* |
| **Technical Requirements and Dependencies**  *Which programming languages, packages, services, and tools will the student need to build this project?*  *Are there any costs or account requirements associated with these tools or services?*  | *Classroom workspace  or local virtual machine with: Security Onion Snort Sguil Wireshark Splunk    Starter files Yoyodyne network diagram Incident Ticket template  Yoyodyne Incident Response Playbooks project.pcap secure.log.gz There are no learner costs associated with this project.* |
| **Project Steps**  *What will the student do to complete the project? Share details about the deliverables and the tools the learner will use.*  ***Note:** these are not detailed project instructions. You'll write those in the next phase when you build the project* *.* | *Use Sguil to review, analyze, and make decisions about Snort/Suricata IDS alerts. Document an incident report. Read packet capture files using tcpdump and analyze them in Wireshark or tshark. Use information from the above analysis to develop a new Snort/Suricata IDS rule to alert on similar network traffic. Install Splunk (Splunk Free edition) on Microsoft Azure. Import log data into Splunk. Search Splunk to identify events related to previously identified security incidents. Recommend mitigation steps. Create a dashboard to identify privilege escalation using sudo. Create a daily report to identify failed logins.* |

